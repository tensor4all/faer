//! Diagnostic probe for the per-process serial mode (not for merge): time `qr_in_place` on a team in one process, and
//! record each member's threadteam trace (publish-to-share latency, park events) and where each thread ran (/proc).
//! Build: `cargo build --release -p t4a-faer --example team_mode_probe --features threadteam/trace`.
//! Run: `taskset -c <cpus> team_mode_probe <width> <n> <reps>`.
//!
//! Options (environment):
//! * `PROBE_PIN=48-63` (or `48,49,..`): pin team participant `i` (the leader is 0) to CPU
//!   `list[i % len]`, with `taskset -pc` from a section on each participant's own thread;
//! * `PROBE_SAMPLE=<cpu>`: a sampler thread, pinned to `<cpu>` (outside the measured set), reads every
//!   thread's state and CPU from `/proc` in a loop during each repetition and reports how often two
//!   runnable (`R`) team threads sat on the same CPU (one of them waiting on that CPU's run queue).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::no_pivoting::factor as qrf;
use faer::utils::team::stats;
use faer::{Mat, Par};
use std::collections::BTreeMap;

fn tasks() -> BTreeMap<u32, (u64, i32, u64, u64)> {
	// tid -> (utime+stime ticks, last cpu, voluntary, involuntary ctx switches)
	let mut m = BTreeMap::new();
	for e in std::fs::read_dir("/proc/self/task").unwrap() {
		let p = e.unwrap().path();
		let tid: u32 = p.file_name().unwrap().to_str().unwrap().parse().unwrap();
		let Ok(stat) = std::fs::read_to_string(p.join("stat")) else { continue };
		let rest = &stat[stat.rfind(')').unwrap() + 2..];
		let f: Vec<&str> = rest.split_whitespace().collect();
		// fields after ")": index 0 = state (field 3); utime = field 14 -> idx 11; stime idx 12; processor field 39 -> idx 36
		let t = f[11].parse::<u64>().unwrap() + f[12].parse::<u64>().unwrap();
		let cpu = f[36].parse::<i32>().unwrap();
		let st = std::fs::read_to_string(p.join("status")).unwrap_or_default();
		let g = |k: &str| st.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0);
		m.insert(tid, (t, cpu, g("voluntary_ctxt_switches:"), g("nonvoluntary_ctxt_switches:")));
	}
	m
}

fn cpu_list(spec: &str) -> Vec<usize> {
	let mut v = Vec::new();
	for part in spec.split(',') {
		if let Some((a, b)) = part.split_once('-') {
			v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap());
		} else {
			v.push(part.parse().unwrap());
		}
	}
	v
}

fn my_tid() -> u32 {
	let l = std::fs::read_link("/proc/thread-self").unwrap();
	l.file_name().unwrap().to_str().unwrap().parse().unwrap()
}

fn pin_tid(tid: u32, cpu: usize) {
	let ok = std::process::Command::new("taskset")
		.args(["-pc", &cpu.to_string(), &tid.to_string()])
		.stdout(std::process::Stdio::null())
		.status()
		.map(|s| s.success())
		.unwrap_or(false);
	assert!(ok, "taskset -pc {cpu} {tid} failed");
}

/// (state, cpu) of every thread of this process except `skip`.
fn states(skip: u32) -> Vec<(u32, char, i32)> {
	let mut v = Vec::new();
	let Ok(rd) = std::fs::read_dir("/proc/self/task") else { return v };
	for e in rd {
		let p = e.unwrap().path();
		let tid: u32 = p.file_name().unwrap().to_str().unwrap().parse().unwrap();
		if tid == skip {
			continue;
		}
		let Ok(stat) = std::fs::read_to_string(p.join("stat")) else { continue };
		let rest = &stat[stat.rfind(')').unwrap() + 2..];
		let f: Vec<&str> = rest.split_whitespace().collect();
		v.push((tid, f[0].chars().next().unwrap(), f[36].parse().unwrap()));
	}
	v
}

fn main() {
	let mut a = std::env::args().skip(1);
	let w: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(4);
	let n: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(512);
	let reps: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(20);
	let a0 = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 101) as f64 / 101.0 + if i == j { 3.0 } else { 0.0 });
	let bs = qrf::recommended_block_size::<f64>(n, n);
	let allowed = std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("Cpus_allowed_list")).unwrap().to_owned();
	let pin = std::env::var("PROBE_PIN").ok().map(|s| cpu_list(&s));
	let builder = threadteam::Pool::builder().threads(w).acquire_timeout(std::time::Duration::from_secs(30));
	// the sampler: (in_rep flag, per-rep (samples, samples with a stacked runnable pair, max runnable))
	use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as O};
	use std::sync::Arc;
	let sampling = Arc::new(AtomicBool::new(false));
	let stop = Arc::new(AtomicBool::new(false));
	let counts = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
	let sampler = std::env::var("PROBE_SAMPLE").ok().map(|c| {
		let cpu: usize = c.parse().unwrap();
		let (sampling, stop, counts) = (sampling.clone(), stop.clone(), counts.clone());
		std::thread::spawn(move || {
			let me = my_tid();
			pin_tid(me, cpu);
			while !stop.load(O::Relaxed) {
				if !sampling.load(O::Relaxed) {
					std::hint::spin_loop();
					continue;
				}
				let st = states(me);
				let mut per_cpu = BTreeMap::new();
				let mut running = 0u64;
				for &(_, state, cpu) in &st {
					if state == 'R' {
						running += 1;
						*per_cpu.entry(cpu).or_insert(0u32) += 1;
					}
				}
				if std::env::var("PROBE_DEBUG").is_ok() && counts[0].load(O::Relaxed) == 5 {
					eprintln!("sample: {st:?}");
				}
				counts[0].fetch_add(1, O::Relaxed);
				if per_cpu.values().any(|&c| c >= 2) {
					counts[1].fetch_add(1, O::Relaxed);
				}
				counts[2].fetch_max(running, O::Relaxed);
			}
		})
	});
	let pool = builder.build().unwrap();
	let team = pool.into_team(w).unwrap();
	let pool_ref = team.pool();
	let mut exec = team.exec(w);

	exec.team(|mut t| {
		// pin every participant of the team (the leader is participant 0) to its own CPU of the
		// list; the pool's spare worker is not a participant and stays where it sleeps
		if let Some(cpus) = pin.clone() {
			t.reborrow().run(|i, _| pin_tid(my_tid(), cpus[i % cpus.len()]));
			println!("pinned: participant i on cpu list[i % len] of {cpus:?}");
		}
		let mut par = Par::Team(t);
		let req = qrf::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default());
		let mut mem = MemBuffer::new(req);
		let mut q = Mat::<f64>::zeros(bs, n);
		let mut times = Vec::new();
		let before = tasks();
		stats::enable(true);
		let _ = stats::take();
		threadteam::trace::clear(pool_ref);
		let mut per_rep = Vec::new();
		let mut stacked = Vec::new();
		for _ in 0..reps {
			let mut m = a0.clone();
			let t0 = threadteam::trace::now_ns(pool_ref);
			for c in counts.iter() {
				c.store(0, O::Relaxed);
			}
			sampling.store(true, O::Relaxed);
			let t = std::time::Instant::now();
			qrf::qr_in_place(m.as_mut(), q.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
			times.push(t.elapsed().as_secs_f64() * 1e3);
			sampling.store(false, O::Relaxed);
			if sampler.is_some() {
				stacked.push((counts[0].load(O::Relaxed), counts[1].load(O::Relaxed), counts[2].load(O::Relaxed)));
			}
			per_rep.push((t0, threadteam::trace::now_ns(pool_ref)));
		}
		let ev = threadteam::trace::events(pool_ref);
		for (r, &(a, b)) in per_rep.iter().enumerate() {
			use threadteam::trace::Kind;
			let inr: Vec<_> = ev.iter().filter(|e| e.ns >= a && e.ns < b).collect();
			let cnt = |k: Kind| inr.iter().filter(|e| e.kind == k).count();
			// per member: share time and latency from the leader's last publish to SectionBegin
			let mut pubs: Vec<u64> = inr.iter().filter(|e| e.kind == Kind::Publish).map(|e| e.ns).collect();
			pubs.sort();
			let mut lat = 0u64; let mut nlat = 0u64; let mut maxlat = 0u64;
			let mut busy = 0u64; let mut open = std::collections::HashMap::new();
			let mut members = std::collections::BTreeSet::new();
			for e in &inr { match e.kind {
				Kind::SectionBegin => { members.insert(e.member); open.insert(e.member, e.ns);
					if let Some(p) = pubs.iter().rev().find(|&&p| p <= e.ns) { let l = e.ns - p; lat += l; nlat += 1; maxlat = maxlat.max(l); } }
				Kind::SectionEnd => if let Some(s) = open.remove(&e.member) { busy += e.ns - s },
				_ => {} } }
			println!("  rep {r}: {:.3} ms publish {} shares {} members {} park {} unpark {} mean_lat {:.1} us max_lat {:.1} us member_busy {:.3} ms",
				times[r], cnt(Kind::Publish), cnt(Kind::SectionBegin), members.len(), cnt(Kind::ParkEnter), cnt(Kind::ParkExit),
				if nlat > 0 { lat as f64 / nlat as f64 / 1e3 } else { 0.0 }, maxlat as f64 / 1e3, busy as f64 / 1e6);
			if let Some(&(n, k, mx)) = stacked.get(r) {
				println!("    sampler: {n} samples, {k} with two runnable threads on one cpu ({:.1} %), at most {mx} runnable", 100.0 * k as f64 / n.max(1) as f64);
			}
		}
		stats::enable(false);
		let s = stats::take();
		let after = tasks();
		let mut sorted = times.clone();
		sorted.sort_by(f64::total_cmp);
		println!("pid {} {allowed} w={w} n={n} median {:.3} ms min {:.3} max {:.3}", std::process::id(), sorted[reps / 2], sorted[0], sorted[reps - 1]);
		println!("  run {:?} split {:?} tiled {} kernel {} below_floor {}", s.run, s.split, s.gemm_tiled, s.gemm_team_kernel, s.gemm_below_floor);
		let mut cpus = BTreeMap::new();
		for (tid, (t1, cpu, v1, i1)) in &after {
			let (t0, _, v0, i0) = before.get(tid).copied().unwrap_or((0, -1, 0, 0));
			if t1 - t0 > 0 {
				println!("  tid {tid} cpu {cpu} ticks {} vol {} invol {}", t1 - t0, v1 - v0, i1 - i0);
				*cpus.entry(*cpu).or_insert(0) += 1;
			}
		}
		println!("  busy threads {} on {} distinct cpus", cpus.values().sum::<i32>(), cpus.len());
	}).unwrap();
	stop.store(true, O::Relaxed);
	if let Some(h) = sampler {
		h.join().unwrap();
	}
}
