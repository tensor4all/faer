//! Probe for tensor4all/faer#107 (not for merge): does the slow mode come with a freshly created pool
//! (new threads), or with every team acquisition?
//!
//! One process times `reps` calls of one kernel at `width` participants. It acquires the team in one
//! of five ways, from "one team held for every call" to "a new pool per call" (what
//! `faer::team::with_team` does, and so `fbench`). After each call, a census section asks every
//! participant which CPU it is on. A call is *stacked* when two participants share a CPU.
//!
//! Build: `cargo build --release -p t4a-faer --example acquire_probe_107`
//! Run:   `taskset -c 0-63 target/release/examples/acquire_probe_107 <mode> <kernel> <width> <reps>`
//!
//! * `mode`:
//!   - `held`: one Pool and Team, one `exec.team` scope around every call (the earlier `team_mode_probe`)
//!   - `team_per_call`: one Pool and Team, a fresh `team.exec(w).team(..)` per call (tenferro `CpuEngine`)
//!   - `exec_team_per_call`: one Pool, `Exec::new(&pool, w).team(..)` per call (members acquired per call)
//!   - `into_team_per_call`: one Pool, `pool.into_team(w)` ... `team.into_pool()` per call
//!   - `fresh_pool`: a new Pool and Team per call, as `with_team` builds them (honours `PIN`)
//!   - `with_team`: `faer::team::with_team(w, ..)` per call, literally
//! * `kernel`: `gemm128` (f64 128x128x128 product) or `chol2048` (f64 LLT of a 2048 SPD matrix)
//!
//! Environment:
//! * `PIN=1`: `PoolBuilder::pin_participants(true)` (every mode except `with_team`)
//! * `PRIME_MS=<ms>`: before the measured calls, run the same mode for this long (unmeasured)
//! * `CENSUS=off`: skip the census section (it is one extra short section per call, after the timing)
//! * `SAMPLE_CPU=<cpu>`: a sampler thread pinned to `<cpu>` (keep it outside the measured set) reads
//!   every thread's state and CPU from `/proc` in a loop while a call runs, and reports per call how
//!   many samples had two runnable (`R`) threads of this process on one CPU
//!
//! Output: one `call` line per call (time, distinct CPUs, stacked), then a `summary` line.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor as llt;
use faer::{Accum, Mat, Par};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use threadteam::{Exec, Pool, Team};

fn this_cpu() -> i32 {
	let stat = std::fs::read_to_string("/proc/thread-self/stat").unwrap_or_default();
	let Some(close) = stat.rfind(')') else { return -1 };
	// fields after ")": index 0 = state (field 3), processor = field 39 -> index 36
	stat[close + 2..].split_whitespace().nth(36).and_then(|v| v.parse().ok()).unwrap_or(-1)
}

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

fn pool(width: usize, pin: bool) -> Pool {
	Pool::builder()
		.threads(width)
		.acquire_timeout(Duration::from_secs(30))
		.pin_participants(pin)
		.build()
		.expect("a pool")
}

/// Every participant reports its CPU.
fn census(team: threadteam::TeamExec<'_>, out: &[AtomicI32]) {
	team.run(|i, _| out[i].store(this_cpu(), Ordering::Relaxed));
}

struct Call {
	ms: f64,
	cpus: Vec<i32>,
	/// (samples with two runnable threads on one CPU, samples) during the call
	stack: (u64, u64),
}

/// The `/proc` sampler: counts, while `on`, samples in which two runnable threads share a CPU.
struct Sampler {
	on: Arc<AtomicBool>,
	stacked: Arc<AtomicU64>,
	total: Arc<AtomicU64>,
}

impl Sampler {
	fn start(cpu: usize) -> Self {
		let on = Arc::new(AtomicBool::new(false));
		let stacked = Arc::new(AtomicU64::new(0));
		let total = Arc::new(AtomicU64::new(0));
		let (o, st, to) = (on.clone(), stacked.clone(), total.clone());
		let (tx, rx) = std::sync::mpsc::channel();
		std::thread::spawn(move || {
			let me = my_tid();
			let ok = std::process::Command::new("taskset")
				.args(["-pc", &cpu.to_string(), &me.to_string()])
				.stdout(std::process::Stdio::null())
				.status()
				.map(|s| s.success())
				.unwrap_or(false);
			assert!(ok, "taskset -pc {cpu} {me} failed");
			tx.send(()).unwrap();
			loop {
				if !o.load(Ordering::Acquire) {
					std::thread::yield_now();
					continue;
				}
				let mut seen = std::collections::BTreeMap::<i32, u32>::new();
				let Ok(rd) = std::fs::read_dir("/proc/self/task") else { continue };
				for e in rd.flatten() {
					let p = e.path();
					if p.file_name().and_then(|f| f.to_str()).and_then(|f| f.parse::<u32>().ok()) == Some(me) {
						continue;
					}
					let Ok(stat) = std::fs::read_to_string(p.join("stat")) else { continue };
					let Some(close) = stat.rfind(')') else { continue };
					let f: Vec<&str> = stat[close + 2..].split_whitespace().collect();
					if f.first() == Some(&"R") {
						if let Some(c) = f.get(36).and_then(|v| v.parse().ok()) {
							*seen.entry(c).or_default() += 1;
						}
					}
				}
				to.fetch_add(1, Ordering::Relaxed);
				if seen.values().any(|&n| n > 1) {
					st.fetch_add(1, Ordering::Relaxed);
				}
			}
		});
		rx.recv().unwrap();
		Sampler { on, stacked, total }
	}

	fn begin(&self) {
		self.stacked.store(0, Ordering::Relaxed);
		self.total.store(0, Ordering::Relaxed);
		self.on.store(true, Ordering::Release);
	}

	fn end(&self) -> (u64, u64) {
		self.on.store(false, Ordering::Release);
		(self.stacked.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
	}
}

fn my_tid() -> u32 {
	std::fs::read_link("/proc/thread-self")
		.ok()
		.and_then(|l| l.file_name().and_then(|f| f.to_str()).and_then(|f| f.parse().ok()))
		.unwrap_or(0)
}

static SAMPLER: std::sync::OnceLock<Option<Sampler>> = std::sync::OnceLock::new();

fn sample_begin() {
	if let Some(Some(s)) = SAMPLER.get() {
		s.begin()
	}
}

fn sample_end() -> (u64, u64) {
	match SAMPLER.get() {
		Some(Some(s)) => s.end(),
		_ => (0, 0),
	}
}

/// One timed call on a team token, then the census on the same team (outside the timing).
fn timed(exec: &mut Exec<'_>, kernel: &mut Kernel, width: usize, census_on: bool) -> Call {
	let slots: Vec<AtomicI32> = (0..width).map(|_| AtomicI32::new(-1)).collect();
	kernel.reset();
	let mut ms = 0.0;
	let mut stack = (0, 0);
	exec.team(|t| {
		sample_begin();
		let t0 = Instant::now();
		kernel.call(Par::Team(t));
		ms = t0.elapsed().as_secs_f64() * 1e3;
		stack = sample_end();
	})
	.ok()
	.expect("a team token");
	if census_on {
		exec.team(|t| census(t, &slots)).ok().expect("a team token");
	}
	Call { ms, cpus: slots.iter().map(|c| c.load(Ordering::Relaxed)).filter(|&c| c >= 0).collect(), stack }
}

fn main() {
	let args: Vec<String> = std::env::args().collect();
	assert!(args.len() == 5, "usage: acquire_probe_107 <mode> <kernel> <width> <reps>");
	let mode = args[1].as_str();
	let mut kernel = Kernel::new(&args[2]);
	let width: usize = args[3].parse().unwrap();
	let reps: usize = args[4].parse().unwrap();
	let pin = std::env::var("PIN").map(|v| v == "1").unwrap_or(false);
	let prime = Duration::from_millis(std::env::var("PRIME_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
	let census_on = std::env::var("CENSUS").map(|v| v != "off").unwrap_or(true);
	SAMPLER.get_or_init(|| std::env::var("SAMPLE_CPU").ok().and_then(|v| v.parse().ok()).map(Sampler::start));

	// the state a mode keeps across calls
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
			"team_per_call" => {
				let team = held_team.as_ref().unwrap();
				timed(&mut team.exec(width), kernel, width, census_on)
			},
			"exec_team_per_call" => {
				let p = held_pool.as_ref().unwrap();
				timed(&mut Exec::new(p, width).unwrap(), kernel, width, census_on)
			},
			"into_team_per_call" => {
				let team = held_pool.take().unwrap().into_team(width).ok().expect("an idle pool");
				let call = timed(&mut team.exec(width), kernel, width, census_on);
				held_pool = Some(team.into_pool());
				call
			},
			"fresh_pool" => {
				let team = pool(width, pin).into_team(width).ok().expect("an idle pool");
				let call = timed(&mut team.exec(width), kernel, width, census_on);
				drop(team);
				call
			},
			"with_team" => {
				let slots: Vec<AtomicI32> = (0..width).map(|_| AtomicI32::new(-1)).collect();
				kernel.reset();
				let mut ms = 0.0;
				let mut stack = (0, 0);
				faer::team::with_team(width, |par| {
					sample_begin();
					let t0 = Instant::now();
					kernel.call(par);
					ms = t0.elapsed().as_secs_f64() * 1e3;
					stack = sample_end();
				});
				// `with_team` gives the team back at the end, so its census needs a team of its own
				if census_on {
					faer::team::with_team(width, |par| {
						if let Par::Team(t) = par {
							census(t, &slots)
						}
					});
				}
				Call { ms, cpus: slots.iter().map(|c| c.load(Ordering::Relaxed)).filter(|&c| c >= 0).collect(), stack }
			},
			_ => unreachable!(),
		}
	};

	if mode == "held" {
		// one scope for the prime and every call, as the earlier probe did
		let team = held_team.as_ref().unwrap();
		let mut exec = team.exec(width);
		let start = Instant::now();
		while start.elapsed() < prime {
			timed(&mut exec, &mut kernel, width, false);
		}
		let calls: Vec<Call> = (0..reps).map(|_| timed(&mut exec, &mut kernel, width, census_on)).collect();
		report(mode, &kernel.name, width, pin, &calls);
		return;
	}
	let start = Instant::now();
	while start.elapsed() < prime {
		one(&mut kernel);
	}
	let calls: Vec<Call> = (0..reps).map(|_| one(&mut kernel)).collect();
	report(mode, &kernel.name, width, pin, &calls);
}

fn report(mode: &str, kernel: &str, width: usize, pin: bool, calls: &[Call]) {
	let mut stacked = 0;
	for (i, c) in calls.iter().enumerate() {
		let mut cpus = c.cpus.clone();
		cpus.sort_unstable();
		cpus.dedup();
		let is_stacked = !c.cpus.is_empty() && cpus.len() < c.cpus.len();
		stacked += is_stacked as usize;
		println!(
			"call mode={mode} kernel={kernel} width={width} pin={pin} i={i} ms={:.4} distinct_cpus={}/{} stacked={is_stacked} stacked_samples={}/{}",
			c.ms,
			cpus.len(),
			c.cpus.len(),
			c.stack.0,
			c.stack.1
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
