//! Probe for tensor4all/faer#107 (not for merge): does the slow mode come with a freshly created pool
//! (new threads), with every team acquisition, or with the pool's spare worker?
//!
//! One process times `reps` calls of one kernel at `width` participants, acquiring the team in one of
//! six ways (below). A pool of `width` workers gives a team of `width` participants (the calling
//! thread and `width - 1` reserved workers), so one worker of the pool is a *spare* that the team
//! never uses (tensor4all/threadteam-rs#12): the process runs `width + 1` threads.
//!
//! After each call, a census section on the same team asks every participant for its CPU and tid. A
//! call is *stacked* when two participants share a CPU. Every other thread of the process, except
//! the caller and the sampler, is a spare worker; for each call the probe reports their CPU time
//! (`/proc/<tid>/schedstat` run time, during the call and since the thread started) and their state
//! at the end of the call.
//!
//! Build: `cargo build --release -p t4a-faer --example acquire_probe_107`
//! Run:   `taskset -c 0-63 target/release/examples/acquire_probe_107 <mode> <kernel> <width> <reps>`
//!
//! * `mode`:
//!   - `held`: one Pool and Team, one `exec.team` scope around every call (the earlier `team_mode_probe`)
//!   - `team_per_call`: one Pool and Team, a fresh `team.exec(w).team(..)` per call (tenferro `CpuEngine`)
//!   - `exec_team_per_call`: one Pool, `Exec::new(&pool, w).team(..)` per call (members acquired per call)
//!   - `into_team_per_call`: one Pool, `pool.into_team(w)` ... `team.into_pool()` per call
//!   - `fresh_pool`: a new Pool and Team per call, built as `with_team` builds them (honours `PIN`)
//!   - `with_team`: `faer::team::with_team(w, ..)` per call, literally
//! * `kernel`: `gemm128` (f64 128x128x128 product) or `chol2048` (f64 LLT of a 2048 SPD matrix)
//!
//! Environment:
//! * `PIN=1`: `PoolBuilder::pin_participants(true)` (every mode except `with_team`)
//! * `PRIME_MS=<ms>`: before the measured calls, run the same mode for this long (unmeasured)
//! * `CENSUS=off`: skip the census section (one extra short section per call, after the timing)
//! * `SAMPLE_CPU=<cpu>`: a sampler thread pinned to `<cpu>` (keep it outside the measured set) reads
//!   every thread's state and CPU from `/proc` in a loop while a call runs, and reports how many
//!   samples had two runnable (`R`) threads of this process on one CPU, how many of those involved a
//!   spare worker, and how many samples found a spare worker runnable at all
//! * `SPARE_WAIT=1` (`fresh_pool`, `with_team`): after each call, before the pool is dropped, wait
//!   (up to 2 s, unmeasured) until every spare worker is sleeping, and report the time from the
//!   pool's creation until then (the spare's spin-until-sleep time)
//!
//! Output: one `call` line per call, then a `summary` line.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor as llt;
use faer::{Accum, Mat, Par};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use threadteam::{Exec, Pool, Team, TeamExec};

// ---------------------------------------------------------------- /proc

fn my_tid() -> u32 {
	std::fs::read_link("/proc/thread-self")
		.ok()
		.and_then(|l| l.file_name().and_then(|f| f.to_str()).and_then(|f| f.parse().ok()))
		.unwrap_or(0)
}

/// (state, cpu) from `/proc/self/task/<tid>/stat`.
fn stat_of(tid: u32) -> Option<(char, i32)> {
	let stat = std::fs::read_to_string(format!("/proc/self/task/{tid}/stat")).ok()?;
	let close = stat.rfind(')')?;
	// fields after ")": index 0 = state (field 3), processor = field 39 -> index 36
	let f: Vec<&str> = stat[close + 2..].split_whitespace().collect();
	Some((f.first()?.chars().next()?, f.get(36)?.parse().ok()?))
}

/// Time on the CPU since the thread started, in ns (`schedstat` field 1).
fn run_ns(tid: u32) -> u64 {
	std::fs::read_to_string(format!("/proc/self/task/{tid}/schedstat"))
		.ok()
		.and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
		.unwrap_or(0)
}

fn tids() -> Vec<u32> {
	let Ok(rd) = std::fs::read_dir("/proc/self/task") else { return Vec::new() };
	rd.flatten().filter_map(|e| e.file_name().to_str().and_then(|f| f.parse().ok())).collect()
}

fn this_cpu() -> i32 {
	stat_of(my_tid()).map(|s| s.1).unwrap_or(-1)
}

// ---------------------------------------------------------------- sampler

#[derive(Default, Clone)]
struct Samples {
	total: u64,
	/// samples with two runnable threads of this process on one CPU
	stacked: u64,
	/// per tid: samples in which it was runnable
	runnable: BTreeMap<u32, u64>,
	/// per tid: samples in which it was runnable on a CPU that had another runnable thread
	stacked_with: BTreeMap<u32, u64>,
}

struct Sampler {
	tid: u32,
	on: Arc<AtomicBool>,
	acc: Arc<Mutex<Samples>>,
}

impl Sampler {
	fn start(cpu: usize) -> Self {
		let on = Arc::new(AtomicBool::new(false));
		let acc = Arc::new(Mutex::new(Samples::default()));
		let tid = Arc::new(AtomicU32::new(0));
		let (o, a, t) = (on.clone(), acc.clone(), tid.clone());
		let (tx, rx) = std::sync::mpsc::channel();
		std::thread::spawn(move || {
			let me = my_tid();
			t.store(me, Ordering::Relaxed);
			let ok = std::process::Command::new("taskset")
				.args(["-pc", &cpu.to_string(), &me.to_string()])
				.stdout(std::process::Stdio::null())
				.status()
				.map(|s| s.success())
				.unwrap_or(false);
			assert!(ok, "taskset -pc {cpu} {me} failed");
			tx.send(()).unwrap();
			let mut by_cpu = BTreeMap::<i32, Vec<u32>>::new();
			loop {
				if !o.load(Ordering::Acquire) {
					std::thread::yield_now();
					continue;
				}
				by_cpu.clear();
				for tid in tids() {
					if tid == me {
						continue;
					}
					if let Some(('R', c)) = stat_of(tid) {
						by_cpu.entry(c).or_default().push(tid);
					}
				}
				let mut s = a.lock().unwrap();
				s.total += 1;
				let mut any = false;
				for list in by_cpu.values() {
					for &tid in list {
						*s.runnable.entry(tid).or_default() += 1;
						if list.len() > 1 {
							*s.stacked_with.entry(tid).or_default() += 1;
						}
					}
					any |= list.len() > 1;
				}
				s.stacked += any as u64;
			}
		});
		rx.recv().unwrap();
		let tid = tid.load(Ordering::Relaxed);
		Sampler { tid, on, acc }
	}
}

static SAMPLER: OnceLock<Option<Sampler>> = OnceLock::new();

fn sampler_tid() -> u32 {
	SAMPLER.get().and_then(|s| s.as_ref()).map(|s| s.tid).unwrap_or(0)
}

fn sample_begin() {
	if let Some(Some(s)) = SAMPLER.get() {
		*s.acc.lock().unwrap() = Samples::default();
		s.on.store(true, Ordering::Release);
	}
}

fn sample_end() -> Samples {
	match SAMPLER.get() {
		Some(Some(s)) => {
			s.on.store(false, Ordering::Release);
			s.acc.lock().unwrap().clone()
		},
		_ => Samples::default(),
	}
}

// ---------------------------------------------------------------- kernels

struct Kernel {
	name: String,
	a: Mat<f64>,
	b: Mat<f64>,
	work: Mat<f64>,
	mem: MemBuffer,
}

impl Kernel {
	fn new(name: &str) -> Self {
		match name {
			"gemm128" => {
				let n = 128;
				Kernel {
					name: name.into(),
					a: Mat::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 - 8.0),
					b: Mat::from_fn(n, n, |i, j| ((i * 5 + j * 3) % 11) as f64 - 5.0),
					work: Mat::zeros(n, n),
					mem: MemBuffer::new(dyn_stack::StackReq::EMPTY),
				}
			},
			"chol2048" => {
				let n = 2048;
				let m = Mat::<f64>::from_fn(n, n, |i, j| (((i * 31 + j * 17) % 23) as f64 - 11.0) / 23.0);
				let mut a = &m * m.transpose();
				for i in 0..n {
					a[(i, i)] += n as f64;
				}
				let mem = MemBuffer::new(llt::cholesky_in_place_scratch::<f64>(n, faer::ParShape::Seq, Default::default()));
				Kernel { name: name.into(), b: Mat::zeros(0, 0), work: a.clone(), a, mem }
			},
			_ => panic!("kernel: gemm128 | chol2048"),
		}
	}

	/// Restores the input (unmeasured).
	fn reset(&mut self) {
		if self.name == "chol2048" {
			self.work.copy_from(&self.a);
		}
	}

	fn call(&mut self, par: Par<'_>) {
		if self.name == "gemm128" {
			faer::linalg::matmul::matmul(self.work.as_mut(), Accum::Replace, self.a.as_ref(), self.b.as_ref(), 1.0, par);
		} else {
			llt::cholesky_in_place(self.work.as_mut(), Default::default(), par, MemStack::new(&mut self.mem), Default::default())
				.unwrap();
		}
	}
}

// ---------------------------------------------------------------- one call

fn pool(width: usize, pin: bool) -> Pool {
	Pool::builder()
		.threads(width)
		.acquire_timeout(Duration::from_secs(30))
		.pin_participants(pin)
		.build()
		.expect("a pool")
}

struct Opts {
	width: usize,
	census: bool,
	spare_wait: bool,
	main_tid: u32,
}

struct Call {
	ms: f64,
	/// (tid, cpu) of every participant, from the census
	parts: Vec<(u32, i32)>,
	samples: Samples,
	/// run time per tid at the start and at the end of the call
	run_begin: BTreeMap<u32, u64>,
	run_end: BTreeMap<u32, u64>,
	/// state per tid at the end of the call
	state_end: BTreeMap<u32, char>,
	/// time from the pool's creation until every spare slept (`SPARE_WAIT`)
	spare_sleep_ms: Option<f64>,
}

fn run_all() -> BTreeMap<u32, u64> {
	tids().into_iter().map(|t| (t, run_ns(t))).collect()
}

/// The timed call and the census, on one team token. `built`: when the pool was created, for
/// `SPARE_WAIT`.
fn timed(team: TeamExec<'_>, kernel: &mut Kernel, o: &Opts, built: Option<Instant>) -> Call {
	let mut team = team;
	let run_begin = run_all();
	sample_begin();
	let t0 = Instant::now();
	kernel.call(Par::Team(team.reborrow()));
	let ms = t0.elapsed().as_secs_f64() * 1e3;
	let samples = sample_end();
	let run_end = run_all();
	let state_end = tids().into_iter().filter_map(|t| stat_of(t).map(|s| (t, s.0))).collect();
	let mut parts = Vec::new();
	if o.census {
		let slots: Vec<(AtomicU32, AtomicI32)> = (0..o.width).map(|_| (AtomicU32::new(0), AtomicI32::new(-1))).collect();
		team.run(|i, _| {
			slots[i].0.store(my_tid(), Ordering::Relaxed);
			slots[i].1.store(this_cpu(), Ordering::Relaxed);
		});
		parts = slots
			.iter()
			.map(|s| (s.0.load(Ordering::Relaxed), s.1.load(Ordering::Relaxed)))
			.filter(|s| s.1 >= 0)
			.collect();
	}
	let mut call = Call { ms, parts, samples, run_begin, run_end, state_end, spare_sleep_ms: None };
	if o.spare_wait {
		if let Some(built) = built {
			let spares = spares(&call, o);
			let deadline = Instant::now() + Duration::from_secs(2);
			while Instant::now() < deadline && spares.iter().any(|&t| !matches!(stat_of(t), Some(('S', _)) | None)) {
				std::thread::yield_now();
			}
			call.spare_sleep_ms = Some(built.elapsed().as_secs_f64() * 1e3);
		}
	}
	call
}

/// Threads that are neither the caller, the sampler nor a participant: the pool's spare workers.
fn spares(c: &Call, o: &Opts) -> Vec<u32> {
	let parts: BTreeSet<u32> = c.parts.iter().map(|p| p.0).collect();
	c.run_end.keys().copied().filter(|t| *t != o.main_tid && *t != sampler_tid() && !parts.contains(t)).collect()
}

fn one_scope(exec: &mut Exec<'_>, kernel: &mut Kernel, o: &Opts, built: Option<Instant>) -> Call {
	kernel.reset();
	exec.team(|t| timed(t, kernel, o, built)).ok().expect("a team token")
}

// ---------------------------------------------------------------- main

fn main() {
	let args: Vec<String> = std::env::args().collect();
	assert!(args.len() == 5, "usage: acquire_probe_107 <mode> <kernel> <width> <reps>");
	let mode = args[1].as_str();
	let mut kernel = Kernel::new(&args[2]);
	let width: usize = args[3].parse().unwrap();
	let reps: usize = args[4].parse().unwrap();
	let env = |k: &str| std::env::var(k).ok();
	let pin = env("PIN").as_deref() == Some("1");
	let prime = Duration::from_millis(env("PRIME_MS").and_then(|v| v.parse().ok()).unwrap_or(0));
	let o = Opts {
		width,
		census: env("CENSUS").as_deref() != Some("off"),
		spare_wait: env("SPARE_WAIT").as_deref() == Some("1"),
		main_tid: my_tid(),
	};
	SAMPLER.get_or_init(|| env("SAMPLE_CPU").and_then(|v| v.parse().ok()).map(Sampler::start));

	let mut held_pool: Option<Pool> = None;
	let mut held_team: Option<Team> = None;
	match mode {
		"held" | "team_per_call" => held_team = Some(pool(width, pin).into_team(width).ok().expect("an idle pool")),
		"exec_team_per_call" | "into_team_per_call" => held_pool = Some(pool(width, pin)),
		"fresh_pool" | "with_team" => {},
		_ => panic!("mode: held | team_per_call | exec_team_per_call | into_team_per_call | fresh_pool | with_team"),
	}

	let mut one = |kernel: &mut Kernel| -> Call {
		match mode {
			"team_per_call" => one_scope(&mut held_team.as_ref().unwrap().exec(width), kernel, &o, None),
			"exec_team_per_call" => one_scope(&mut Exec::new(held_pool.as_ref().unwrap(), width).unwrap(), kernel, &o, None),
			"into_team_per_call" => {
				let team = held_pool.take().unwrap().into_team(width).ok().expect("an idle pool");
				let call = one_scope(&mut team.exec(width), kernel, &o, None);
				held_pool = Some(team.into_pool());
				call
			},
			"fresh_pool" => {
				let built = Instant::now();
				let team = pool(width, pin).into_team(width).ok().expect("an idle pool");
				let call = one_scope(&mut team.exec(width), kernel, &o, Some(built));
				drop(team);
				call
			},
			"with_team" => {
				kernel.reset();
				let built = Instant::now();
				faer::team::with_team(width, |par| match par {
					Par::Team(t) => timed(t, kernel, &o, Some(built)),
					_ => panic!("with_team gave no team"),
				})
			},
			_ => unreachable!(),
		}
	};

	let calls: Vec<Call> = if mode == "held" {
		// one scope for the prime and every call, as the earlier probe did
		let team = held_team.as_ref().unwrap();
		let mut exec = team.exec(width);
		exec.team(|mut t| {
			let start = Instant::now();
			let quiet = Opts { census: false, spare_wait: false, ..o };
			while start.elapsed() < prime {
				kernel.reset();
				timed(t.reborrow(), &mut kernel, &quiet, None);
			}
			let o = Opts { ..quiet };
			let o = Opts { census: env("CENSUS").as_deref() != Some("off"), ..o };
			(0..reps)
				.map(|_| {
					kernel.reset();
					timed(t.reborrow(), &mut kernel, &o, None)
				})
				.collect()
		})
		.ok()
		.expect("a team token")
	} else {
		let start = Instant::now();
		while start.elapsed() < prime {
			one(&mut kernel);
		}
		(0..reps).map(|_| one(&mut kernel)).collect()
	};
	report(mode, &kernel.name, width, pin, &o, &calls);
}

fn report(mode: &str, kernel: &str, width: usize, pin: bool, o: &Opts, calls: &[Call]) {
	let mut stacked = 0;
	for (i, c) in calls.iter().enumerate() {
		let cpus: BTreeSet<i32> = c.parts.iter().map(|p| p.1).collect();
		let is_stacked = !c.parts.is_empty() && cpus.len() < c.parts.len();
		stacked += is_stacked as usize;
		let spare = if c.parts.is_empty() { Vec::new() } else { spares(c, o) };
		let us = |m: &BTreeMap<u32, u64>| spare.iter().map(|t| m.get(t).copied().unwrap_or(0)).sum::<u64>() as f64 / 1e3;
		let in_call = us(&c.run_end) - us(&c.run_begin);
		let states: String = spare.iter().map(|t| c.state_end.get(t).copied().unwrap_or('?')).collect();
		let sum = |m: &BTreeMap<u32, u64>| spare.iter().map(|t| m.get(t).copied().unwrap_or(0)).sum::<u64>();
		println!(
			"call mode={mode} kernel={kernel} width={width} pin={pin} i={i} ms={:.4} distinct_cpus={}/{} stacked={is_stacked} \
			 stacked_samples={}/{} spares={} spare_state_end={} spare_run_us_call={:.1} spare_run_us_total={:.1} \
			 spare_runnable_samples={} spare_stacked_samples={} spare_sleep_ms={}",
			c.ms,
			cpus.len(),
			c.parts.len(),
			c.samples.stacked,
			c.samples.total,
			spare.len(),
			if states.is_empty() { "-".into() } else { states },
			in_call,
			us(&c.run_end),
			sum(&c.samples.runnable),
			sum(&c.samples.stacked_with),
			c.spare_sleep_ms.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
		);
	}
	let mut ms: Vec<f64> = calls.iter().map(|c| c.ms).collect();
	ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
	println!(
		"summary mode={mode} kernel={kernel} width={width} pin={pin} reps={} min_ms={:.4} median_ms={:.4} max_ms={:.4} stacked_calls={stacked}",
		ms.len(),
		ms[0],
		ms[ms.len() / 2],
		ms[ms.len() - 1]
	);
}
