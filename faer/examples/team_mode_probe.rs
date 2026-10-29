//! Diagnostic probe for the per-process serial mode (not for merge): time `qr_in_place` on a team in one process, and
//! record each member's threadteam trace (publish-to-share latency, park events) and where each thread ran (/proc).
//! Build: `cargo build --release -p t4a-faer --example team_mode_probe --features threadteam/trace`.
//! Run: `taskset -c <cpus> team_mode_probe <width> <n> <reps>`.
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

fn main() {
	let mut a = std::env::args().skip(1);
	let w: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(4);
	let n: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(512);
	let reps: usize = a.next().map(|s| s.parse().unwrap()).unwrap_or(20);
	let a0 = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 101) as f64 / 101.0 + if i == j { 3.0 } else { 0.0 });
	let bs = qrf::recommended_block_size::<f64>(n, n);
	let allowed = std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("Cpus_allowed_list")).unwrap().to_owned();
	let pool = threadteam::Pool::builder().threads(w).acquire_timeout(std::time::Duration::from_secs(30)).build().unwrap();
	let team = pool.into_team(w).unwrap();
	let pool_ref = team.pool();
	let mut exec = team.exec(w);

	exec.team(|t| { let mut par = Par::Team(t);
		let req = qrf::qr_in_place_scratch::<f64>(n, n, bs, par.shape(), Default::default());
		let mut mem = MemBuffer::new(req);
		let mut q = Mat::<f64>::zeros(bs, n);
		let mut times = Vec::new();
		let before = tasks();
		stats::enable(true);
		let _ = stats::take();
		threadteam::trace::clear(pool_ref);
		let mut per_rep = Vec::new();
		for _ in 0..reps {
			let mut m = a0.clone();
			let t0 = threadteam::trace::now_ns(pool_ref);
			let t = std::time::Instant::now();
			qrf::qr_in_place(m.as_mut(), q.as_mut(), par.reborrow(), MemStack::new(&mut mem), Default::default());
			times.push(t.elapsed().as_secs_f64() * 1e3);
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
}
