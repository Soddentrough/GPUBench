#![windows_subsystem = "windows"]

use iced::widget::{button, column, container, progress_bar, row, scrollable, text, Space, tooltip};
use iced::{color, Background, Border, Command, Element, Length, Theme, executor, Application, Settings};
use gpubench_core::{get_available_benchmarks, run_benchmarks, ResultData};
use std::sync::{mpsc, mpsc::Sender, Mutex, LazyLock};
use std::collections::HashSet;

struct PrimaryGradientButton;
impl iced::widget::button::StyleSheet for PrimaryGradientButton {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        use iced::gradient::Linear;
        let mut gradient = Linear::new(0.0);
        gradient = gradient.add_stop(0.0, color!(0x1A2980, 0.85));
        gradient = gradient.add_stop(1.0, color!(0x26D0CE, 0.85));
        iced::widget::button::Appearance {
            background: Some(Background::Gradient(gradient.into())),
            text_color: color!(0xFFFFFF),
            border: Border { radius: 25.0.into(), width: 0.0, color: color!(0x000000, 0.0) },
            shadow_offset: iced::Vector::new(0.0, 4.0),
            ..Default::default()
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        let mut app = self.active(style);
        use iced::gradient::Linear;
        let mut gradient = Linear::new(0.0);
        gradient = gradient.add_stop(0.0, color!(0x1A2980, 1.0));
        gradient = gradient.add_stop(1.0, color!(0x26D0CE, 1.0));
        app.background = Some(Background::Gradient(gradient.into()));
        app
    }
}

struct PillToggle {
    is_active: bool,
    is_api_selector: bool,
}

impl iced::widget::button::StyleSheet for PillToggle {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            if self.is_api_selector {
                iced::widget::button::Appearance {
                    background: Some(Background::Color(color!(0x00E5FF, 0.1))),
                    text_color: color!(0x00E5FF),
                    border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x00E5FF, 0.8) },
                    ..Default::default()
                }
            } else {
                iced::widget::button::Appearance {
                    background: Some(Background::Color(color!(0xFFFFFF, 0.1))),
                    text_color: color!(0xFFFFFF),
                    border: Border { radius: 12.0.into(), width: 1.0, color: color!(0xFFFFFF, 0.6) },
                    ..Default::default()
                }
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x1A1A24))),
                text_color: color!(0x666677),
                border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x1A1A24) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            self.active(style)
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x222233))),
                text_color: color!(0xAAAAAA),
                border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x333344) },
                ..Default::default()
            }
        }
    }
}

struct GroupPill {
    is_highlighted: bool,
}

impl iced::widget::button::StyleSheet for GroupPill {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_highlighted {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0xFF3366, 0.15))),
                text_color: color!(0xFF3366),
                border: Border { radius: 20.0.into(), width: 1.0, color: color!(0xFF3366, 0.6) },
                ..Default::default()
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x1A1A24))),
                text_color: color!(0x8888AA),
                border: Border { radius: 20.0.into(), width: 1.0, color: color!(0x222233) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        iced::widget::button::Appearance {
            background: Some(Background::Color(color!(0xFF3366, 0.1))),
            text_color: color!(0xFF3366),
            border: Border { radius: 20.0.into(), width: 1.0, color: color!(0xFF3366, 0.3) },
            ..Default::default()
        }
    }
}

pub fn main() -> iced::Result {
    GPUBenchApp::run(Settings {
        antialiasing: true,
        window: iced::window::Settings {
            size: iced::Size::new(980.0, 700.0),
            ..Default::default()
        },
        ..Settings::default()
    })
}

// Use a Mutex around the Option instead of OnceLock to allow resetting the sender
static PROGRESS_SENDER: LazyLock<Mutex<Option<Sender<ResultData>>>> = LazyLock::new(|| Mutex::new(None));

fn progress_callback(res: &ResultData) {
    if let Ok(guard) = PROGRESS_SENDER.lock() {
        if let Some(sender) = guard.as_ref() {
            let _ = sender.send(res.clone());
        }
    }
}

/// Extract just the product name from a hardware string like "vulkan|0|AMD Radeon AI PRO R9700 (RADV GFX1201)"
/// Strips driver info in parentheses and common prefixes.
fn clean_device_name(raw: &str) -> String {
    let name = raw.to_string();
    // Strip parenthesized driver info like "(RADV GFX1201)" or "(TM)"
    let mut cleaned = String::new();
    let mut depth = 0i32;
    for c in name.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 => cleaned.push(c),
            _ => {}
        }
    }
    cleaned.trim().to_string()
}

enum AppState {
    Setup {
        available_backends: Vec<String>,
        selected_backend: String,
        
        available_devices: Vec<String>,
        selected_device: String,
        
        available_tests: Vec<String>,
    },
    Running {
        progress_receiver: Option<mpsc::Receiver<ResultData>>,
        total_benchmarks: usize,
        completed_suites: std::collections::HashSet<String>,
    },
    Complete {
        total_benchmarks: usize,
    },
}

struct GPUBenchApp {
    state: AppState,
    current_benchmark: String,
    current_device: String,
    selected_tests: HashSet<String>,
    
    // Metrics
    gpu_bw: f32,
    cpu_bw: f32,
    
    sys_mem_bw: f32,
    sys_mem_bw_single: f32,
    sys_mem_lat: f32,
    
    gpu_fp64: f32,
    gpu_fp32: f32,
    gpu_fp16_vector: f32,
    gpu_fp16_matrix: f32,
    gpu_bf16_vector: f32,
    gpu_bf16_matrix: f32,
    gpu_fp8_vector: f32,
    gpu_fp8_matrix: f32,
    gpu_int8_vector: f32,
    gpu_int8_matrix: f32,
    gpu_int4_vector: f32,
    gpu_int4_matrix: f32,
    
    gpu_rt_anyhit: f32,
    gpu_rt_as_build: f32,
    gpu_rt_incoherent: f32,
    gpu_rt_intersect: f32,
    gpu_rt_divergence: f32,
    gpu_rt_payload: f32,
    gpu_rt_procedural: f32,
}

#[derive(Debug, Clone)]
enum Message {
    BackendSelected(String),
    DeviceSelected(String),
    TestToggled(String, bool),
    TestGroupSelected(String),
    StartBenchmarks,
    BenchmarksComplete,
    Tick,
    SaveResults,
}

impl Application for GPUBenchApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let tests = get_available_benchmarks();
        
        let backends = vec!["VULKAN".to_string(), "ROCm".to_string()];
        let selected_backend = "VULKAN".to_string();
        
        let hw = gpubench_core::get_available_hardware();
        let mut devices = Vec::new();
        for h in &hw {
            let parts: Vec<&str> = h.split('|').collect();
            if parts.len() == 3 {
                let api = parts[0];
                let is_match = if selected_backend == "VULKAN" { api == "vulkan" } else { api == "opencl" };
                if is_match {
                    let cleaned = clean_device_name(parts[2]);
                    devices.push(format!("{}: {}", parts[1], cleaned));
                }
            }
        }
        if devices.is_empty() {
            devices.push("0: Default Device".to_string());
        }

        let selected_device = devices[0].clone();
        
        let mut initial_tests = HashSet::new();
        for t in &tests {
            initial_tests.insert(t.clone());
        }

        (
            Self {
                state: AppState::Setup {
                    available_backends: backends,
                    selected_backend,
                    available_devices: devices,
                    selected_device,
                    available_tests: tests,
                },
                selected_tests: initial_tests,
                current_benchmark: String::from("Waiting to start..."),
                current_device: String::from(""),
                gpu_bw: 0.0,
                cpu_bw: 0.0,
                sys_mem_bw: 0.0,
                sys_mem_bw_single: 0.0,
                sys_mem_lat: 0.0,
                gpu_fp64: 0.0,
                gpu_fp32: 0.0,
                gpu_fp16_vector: 0.0,
                gpu_fp16_matrix: 0.0,
                gpu_bf16_vector: 0.0,
                gpu_bf16_matrix: 0.0,
                gpu_fp8_vector: 0.0,
                gpu_fp8_matrix: 0.0,
                gpu_int8_vector: 0.0,
                gpu_int8_matrix: 0.0,
                gpu_int4_vector: 0.0,
                gpu_int4_matrix: 0.0,
                gpu_rt_anyhit: 0.0,
                gpu_rt_as_build: 0.0,
                gpu_rt_incoherent: 0.0,
                gpu_rt_intersect: 0.0,
                gpu_rt_divergence: 0.0,
                gpu_rt_payload: 0.0,
                gpu_rt_procedural: 0.0,
            },
            Command::none()
        )
    }

    fn title(&self) -> String {
        String::from("BenchmarkX 2026 - Pro Edition")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::BackendSelected(backend) => {
                if let AppState::Setup { selected_backend, available_devices, selected_device, .. } = &mut self.state {
                    *selected_backend = backend.clone();
                    
                    let hw = gpubench_core::get_available_hardware();
                    let mut new_devices = Vec::new();
                    for h in &hw {
                        let parts: Vec<&str> = h.split('|').collect();
                        if parts.len() == 3 {
                            let api = parts[0];
                            let is_match = if *selected_backend == "VULKAN" { api == "vulkan" } else { api == "opencl" };
                            if is_match {
                                let cleaned = clean_device_name(parts[2]);
                                new_devices.push(format!("{}: {}", parts[1], cleaned));
                            }
                        }
                    }
                    if new_devices.is_empty() {
                        new_devices.push("0: Default Device".to_string());
                    }
                    *available_devices = new_devices.clone();
                    *selected_device = new_devices[0].clone();
                }
            }
            Message::DeviceSelected(device) => {
                if let AppState::Setup { selected_device, .. } = &mut self.state {
                    *selected_device = device;
                }
            }
            Message::TestToggled(name, is_checked) => {
                if is_checked {
                    self.selected_tests.insert(name);
                } else {
                    self.selected_tests.remove(&name);
                }
            }
            Message::TestGroupSelected(group) => {
                if let AppState::Setup { available_tests, .. } = &mut self.state {
                    match group.as_str() {
                        "ALL" => {
                            for t in available_tests.iter() {
                                self.selected_tests.insert(t.clone());
                            }
                        }
                        "NONE" => {
                            self.selected_tests.clear();
                        }
                        "COMPUTE" => {
                            let compute: Vec<String> = available_tests.iter()
                                .filter(|t| !t.starts_with("Ray") && *t != "MemBandwidth")
                                .cloned().collect();
                            let global_all = available_tests.iter().all(|t| self.selected_tests.contains(t));
                            if global_all {
                                self.selected_tests.clear();
                                for t in compute { self.selected_tests.insert(t); }
                            } else {
                                let all_selected = compute.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &compute { self.selected_tests.remove(t); }
                                } else {
                                    for t in compute { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        "SYSTEM" => {
                            let sys: Vec<String> = available_tests.iter()
                                .filter(|t| *t == "MemBandwidth")
                                .cloned().collect();
                            let global_all = available_tests.iter().all(|t| self.selected_tests.contains(t));
                            if global_all {
                                self.selected_tests.clear();
                                for t in sys { self.selected_tests.insert(t); }
                            } else {
                                let all_selected = sys.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &sys { self.selected_tests.remove(t); }
                                } else {
                                    for t in sys { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        "RAY TRACING" => {
                            let rt: Vec<String> = available_tests.iter()
                                .filter(|t| t.starts_with("Ray"))
                                .cloned().collect();
                            let global_all = available_tests.iter().all(|t| self.selected_tests.contains(t));
                            if global_all {
                                self.selected_tests.clear();
                                for t in rt { self.selected_tests.insert(t); }
                            } else {
                                let all_selected = rt.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &rt { self.selected_tests.remove(t); }
                                } else {
                                    for t in rt { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::StartBenchmarks => {
                if let AppState::Setup { selected_backend, selected_device, .. } = &self.state {
                    let b_str = selected_backend.clone();
                    let d_idx: u32 = selected_device.split(':').next().unwrap_or("0").parse().unwrap_or(0);
                    self.current_device = selected_device.clone();
                    let tests_to_run: Vec<String> = self.selected_tests.iter().cloned().collect();
                    let total = tests_to_run.len();
                    if total == 0 { return Command::none(); }
                    let (tx, rx) = mpsc::channel();
                    
                    if let Ok(mut guard) = PROGRESS_SENDER.lock() {
                        *guard = Some(tx);
                    }

                    self.state = AppState::Running {
                        progress_receiver: Some(rx),
                        total_benchmarks: total,
                        completed_suites: std::collections::HashSet::new(),
                    };

                    return Command::batch(vec![
                        Command::perform(
                            async { tokio::time::sleep(std::time::Duration::from_millis(50)).await; },
                            |_| Message::Tick
                        ),
                        Command::perform(
                            async move {
                                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                                let res = tokio::task::spawn_blocking(move || {
                                    run_benchmarks(
                                        &tests_to_run,
                                        &vec![d_idx],
                                        &vec![b_str],
                                        false,
                                        false,
                                        false,
                                        progress_callback
                                    )
                                }).await;
                            },
                            |_| Message::BenchmarksComplete
                        )
                    ]);
                }
            }
            Message::Tick => {
                let mut results = Vec::new();
                if let AppState::Running { progress_receiver, .. } = &self.state {
                    if let Some(rx) = progress_receiver.as_ref() {
                        while let Ok(res) = rx.try_recv() {
                            results.push(res);
                        }
                    }
                }
                
                let mut results_to_process = Vec::new();
                for res in results {
                    results_to_process.push(res);
                }

                if let AppState::Running { completed_suites, .. } = &mut self.state {
                    for res in &results_to_process {
                        completed_suites.insert(res.component.clone());
                    }
                }
                
                for res in results_to_process {
                    self.process_result(&res);
                }

                if matches!(self.state, AppState::Running { .. }) {
                    return Command::perform(
                        async { tokio::time::sleep(std::time::Duration::from_millis(50)).await; },
                        |_| Message::Tick
                    );
                }
            }
            Message::BenchmarksComplete => {
                let mut results_to_process = Vec::new();
                if let AppState::Running { ref mut progress_receiver, ref mut completed_suites, .. } = self.state {
                    if let Some(rx) = progress_receiver.take() {
                        while let Ok(res) = rx.try_recv() {
                            completed_suites.insert(res.component.clone());
                            results_to_process.push(res);
                        }
                    }
                }
                for res in results_to_process {
                    self.process_result(&res);
                }
                let mut tb = 1;
                if let AppState::Running { total_benchmarks, .. } = self.state {
                    tb = total_benchmarks;
                }
                self.state = AppState::Complete { total_benchmarks: tb };
                self.current_benchmark = String::from("");
                return Command::none();
            }
            Message::SaveResults => {
                if matches!(self.state, AppState::Complete { .. }) {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("JSON File", &["json"])
                        .set_file_name("gpubench_results.json")
                        .save_file() {
                        
                        let data = serde_json::json!({
                            "hardware": self.current_device,
                            "compute_api": "Vulkan",
                            "results": {
                                "compute": {
                                    "fp64_tflops": self.gpu_fp64,
                                    "fp32_tflops": self.gpu_fp32,
                                    "fp16_vector_tflops": self.gpu_fp16_vector,
                                    "fp16_matrix_tflops": self.gpu_fp16_matrix,
                                    "bf16_vector_tflops": self.gpu_bf16_vector,
                                    "bf16_matrix_tflops": self.gpu_bf16_matrix,
                                    "fp8_vector_tflops": self.gpu_fp8_vector,
                                    "fp8_matrix_tflops": self.gpu_fp8_matrix,
                                    "int8_vector_tops": self.gpu_int8_vector,
                                    "int8_matrix_tops": self.gpu_int8_matrix,
                                    "int4_vector_tops": self.gpu_int4_vector,
                                    "int4_matrix_tops": self.gpu_int4_matrix,
                                },
                                "memory": {
                                    "bandwidth_gbps": self.gpu_bw,
                                },
                                "ray_tracing": {
                                    "any_hit": self.gpu_rt_anyhit,
                                    "as_build": self.gpu_rt_as_build,
                                    "incoherent": self.gpu_rt_incoherent,
                                    "intersect": self.gpu_rt_intersect,
                                    "divergence": self.gpu_rt_divergence,
                                    "payload": self.gpu_rt_payload,
                                    "procedural": self.gpu_rt_procedural,
                                }
                            }
                        });
                        if let Ok(json_str) = serde_json::to_string_pretty(&data) {
                            let _ = std::fs::write(path, json_str);
                        }
                    }
                }
                return Command::none();
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let (status_text, status_color) = match &self.state {
            AppState::Setup { .. } => ("SYSTEM IDLE", color!(0x555566)),
            AppState::Running { .. } => ("ENGAGED", color!(0x00FF88)),
            AppState::Complete { .. } => ("COMPLETE", color!(0x00E5FF)),
        };

        // Header integrated into sidebars

        match &self.state {
            AppState::Setup { available_backends, selected_backend, available_devices, selected_device, available_tests } => {
                let mut device_col = column![].spacing(8);
                for dev in available_devices {
                    let is_sel = dev == selected_device;
                    let dev_btn = button(text(dev).size(13).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([14, 16])
                        .width(Length::Fill)
                        .on_press(Message::DeviceSelected(dev.clone()))
                        .style(iced::theme::Button::Custom(Box::new(PillToggle { is_active: is_sel, is_api_selector: false })));
                    device_col = device_col.push(dev_btn);
                }

                let mut api_col = column![].spacing(8);
                for api in available_backends {
                    let is_sel = api == selected_backend;
                    let api_btn = button(text(api).size(13).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([14, 16])
                        .width(Length::Fill)
                        .on_press(Message::BackendSelected(api.clone()))
                        .style(iced::theme::Button::Custom(Box::new(PillToggle { is_active: is_sel, is_api_selector: true })));
                    api_col = api_col.push(api_btn);
                }

                let start_btn = button(
                    container(text("BEGIN BENCHMARK").size(16).style(color!(0xFFFFFF)))
                        .width(Length::Fill)
                        .center_x()
                )
                .width(Length::Fill)
                .padding([18, 0])
                .on_press(Message::StartBenchmarks)
                .style(iced::theme::Button::Custom(Box::new(PrimaryGradientButton)));

                let sidebar = container(
                    column![
                        text("BENCHMARK").size(28).style(color!(0xFFFFFF)),
                        text("X PRO").size(28).style(color!(0x00E5FF)),
                        Space::with_height(50),
                        text("TARGET HARDWARE").size(11).style(color!(0x8888AA)),
                        Space::with_height(10),
                        device_col,
                        Space::with_height(40),
                        text("COMPUTE API").size(11).style(color!(0x8888AA)),
                        Space::with_height(10),
                        api_col,
                        Space::with_height(Length::Fill),
                        start_btn
                    ]
                )
                .width(Length::Fixed(300.0))
                .height(Length::Fill)
                .padding(30)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0A0F))),
                    border: Border { color: color!(0x1A1A24), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let create_pill_grid = |title: &str, items: Vec<&str>| {
                    let mut col = column![text(title).size(14).style(color!(0xFFFFFF)), Space::with_height(5)].spacing(12);
                    let mut current_row = row![].spacing(12);
                    let mut count = 0;
                    for t in items {
                        if available_tests.contains(&t.to_string()) {
                            let is_checked = self.selected_tests.contains(t);
                            let name = t.to_string();
                            let pill = button(text(t).size(13).horizontal_alignment(iced::alignment::Horizontal::Center))
                                .padding([12, 0])
                                .width(Length::Fill)
                                .on_press(Message::TestToggled(name.clone(), !is_checked))
                                .style(iced::theme::Button::Custom(Box::new(PillToggle { is_active: is_checked, is_api_selector: false })));
                            
                            current_row = current_row.push(pill);
                            count += 1;
                            if count % 2 == 0 {
                                col = col.push(current_row);
                                current_row = row![].spacing(12);
                            }
                        }
                    }
                    if count % 2 != 0 {
                        col = col.push(current_row.push(Space::with_width(Length::Fill)));
                    }
                    container(col).padding(20).style(|_t: &Theme| container::Appearance {
                        background: Some(Background::Color(color!(0x12121A))),
                        border: Border { radius: 16.0.into(), width: 1.0, color: color!(0x1E1E28) },
                        ..Default::default()
                    })
                };

                let comp_col = create_pill_grid("COMPUTE", vec!["FP64", "FP32", "FP16", "BF16", "FP8", "INT8", "INT4"]);
                let sys_col = create_pill_grid("SYSTEM", vec!["MemBandwidth", "SysMemBandwidth", "SysMemLatency"]);
                let rt_col = create_pill_grid("RAY TRACING", vec!["RayTracing", "RayDivergence", "RayAnyHit", "RayIncoherent", "RayPayload", "RayASBuild", "RayProcedural"]);

                let compute_tests: Vec<&String> = available_tests.iter().filter(|t| !t.starts_with("Ray") && !t.contains("MemBandwidth") && !t.contains("SysMem")).collect();
                let sys_tests: Vec<&String> = available_tests.iter().filter(|t| t.contains("MemBandwidth") || t.contains("SysMem")).collect();
                let rt_tests: Vec<&String> = available_tests.iter().filter(|t| t.starts_with("Ray")).collect();
                let all_selected = available_tests.iter().all(|t| self.selected_tests.contains(t));
                let none_selected = self.selected_tests.is_empty();
                let compute_all = compute_tests.iter().all(|t| self.selected_tests.contains(*t));
                let sys_all = sys_tests.iter().all(|t| self.selected_tests.contains(*t));
                let rt_all = rt_tests.iter().all(|t| self.selected_tests.contains(*t));

                let group_toggles = row![
                    button(text("All").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([6, 16])
                        .on_press(Message::TestGroupSelected("ALL".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(GroupPill { is_highlighted: all_selected }))),
                    button(text("None").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([6, 16])
                        .on_press(Message::TestGroupSelected("NONE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(GroupPill { is_highlighted: none_selected }))),
                    button(text("Compute").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([6, 16])
                        .on_press(Message::TestGroupSelected("COMPUTE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(GroupPill { is_highlighted: compute_all && !all_selected }))),
                    button(text("System").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([6, 16])
                        .on_press(Message::TestGroupSelected("SYSTEM".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(GroupPill { is_highlighted: sys_all && !all_selected }))),
                    button(text("Ray Tracing").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([6, 16])
                        .on_press(Message::TestGroupSelected("RAY TRACING".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(GroupPill { is_highlighted: rt_all && !all_selected }))),
                ].spacing(8);

                let main_area = container(
                    scrollable(
                        column![
                            row![
                                text("TEST WORKLOADS").size(24).style(color!(0xFFFFFF)),
                                Space::with_width(Length::Fill),
                                group_toggles
                            ].align_items(iced::Alignment::Center),
                            Space::with_height(30),
                            row![
                                comp_col.width(Length::FillPortion(1)),
                                column![sys_col, rt_col].spacing(20).width(Length::FillPortion(1))
                            ].spacing(20)
                        ]
                    ).height(Length::Fill)
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(40)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x111116))),
                    ..Default::default()
                });

                row![sidebar, main_area].width(Length::Fill).height(Length::Fill).into()
            }
            state @ AppState::Running { .. } | state @ AppState::Complete { .. } => {
                let (total, completed) = match state {
                    AppState::Running { total_benchmarks, completed_suites, .. } => (*total_benchmarks as f32, completed_suites.len() as f32),
                    AppState::Complete { total_benchmarks } => (*total_benchmarks as f32, *total_benchmarks as f32),
                    _ => (1.0, 0.0),
                };

                let global_progress = column![
                    row![text("SUITE PROGRESS").size(14).style(color!(0x8888AA)), Space::with_width(Length::Fill), text(format!("{}/{}", completed, total)).size(14).style(color!(0x8888AA))],
                    progress_bar(0.0..=total.max(completed.max(1.0)), completed).height(8.0),
                ].spacing(8);

                let metric_row = |key: &str, label: &str, val: f32, unit: &str, desc: &str| -> Element<'_, Message> {
                    let is_active = self.selected_tests.contains(key);
                    
                    let val_str = if !is_active {
                        String::from("")
                    } else if val <= 0.0 {
                        if matches!(self.state, AppState::Complete { .. }) {
                            String::from("UNSUPPORTED")
                        } else {
                            String::from("PENDING")
                        }
                    } else if val < 10.0 {
                        format!("{:.2} {}", val, unit)
                    } else {
                        format!("{:.1} {}", val, unit)
                    };

                    let text_color = if !is_active { color!(0x333344) } else if val <= 0.0 { color!(0x444455) } else { color!(0xFFFFFF) };
                    
                    let label_with_tooltip = row![
                        text(label).size(14).style(if val <= 0.0 { color!(0x666677) } else { color!(0xAAAAAA) }),
                        Space::with_width(4),
                        tooltip(
                            text("(?)").size(12).style(color!(0x8888AA)),
                            text(desc).size(12),
                            tooltip::Position::Right
                        )
                        .gap(4)
                        .style(iced::theme::Container::Box)
                    ].align_items(iced::Alignment::Center);

                    column![
                        row![
                            label_with_tooltip, 
                            Space::with_width(Length::Fill), 
                            text(val_str).size(14).style(text_color)
                        ],
                        Space::with_height(4) // Minimal spacing instead of the bar
                    ].spacing(6).into()
                };

                let sys_content = column![
                    metric_row("MemBandwidth", "GPU VRAM Bandwidth", self.gpu_bw, "GB/s", "Measures the maximum rate at which data can be read from or stored into the GPU's VRAM. Critical for high-resolution textures and large datasets."),
                    metric_row("SysMemBandwidth", "System RAM Bandwidth", self.sys_mem_bw, "GB/s", "Measures the maximum multi-threaded bandwidth to the host's system RAM. Important for CPU-to-GPU data transfers and general system performance."),
                    metric_row("SysMemBandwidth", "System RAM (1 Thread)", self.sys_mem_bw_single, "GB/s", "Measures single-threaded bandwidth to system RAM, which indicates memory channel efficiency and latency-bound transfer speeds."),
                    metric_row("SysMemLatency", "System RAM Latency", self.sys_mem_lat, "ns", "Measures the time it takes to fetch a single un-cached piece of data from system memory. Lower is better. Essential for game engines and unpredictable data access."),
                ].spacing(12).into();

                let compute_content = column![
                    metric_row("FP64", "FP64 (Vector)", self.gpu_fp64, "TFLOPS", "Measures double precision (64-bit) floating point operations per second. Crucial for scientific simulations and high-accuracy physics."),
                    metric_row("FP32", "FP32 (Vector)", self.gpu_fp32, "TFLOPS", "Measures single precision (32-bit) floating point operations per second. The standard metric for generic gaming and graphics compute workloads."),
                    metric_row("FP16", "FP16 (Vector)", self.gpu_fp16_vector, "TFLOPS", "Measures vector half precision (16-bit) floating point operations per second. Used extensively in modern rendering, mobile ML, and HDR imaging."),
                    metric_row("FP16", "FP16 (Matrix)", self.gpu_fp16_matrix, "TFLOPS", "Measures hardware-accelerated cooperative matrix half precision (16-bit) operations."),
                    metric_row("BF16", "BF16 (Vector)", self.gpu_bf16_vector, "TFLOPS", "Measures vector Brain Float 16 operations per second. Primarily utilized in AI training and deep learning models to retain dynamic range while saving bandwidth."),
                    metric_row("BF16", "BF16 (Matrix)", self.gpu_bf16_matrix, "TFLOPS", "Measures hardware-accelerated cooperative matrix Brain Float 16 operations."),
                    metric_row("FP8", "FP8 (Vector)", self.gpu_fp8_vector, "TFLOPS", "Measures vector quarter precision (8-bit) floating point operations per second. Used for highly optimized AI inference where memory bandwidth is the primary bottleneck."),
                    metric_row("FP8", "FP8 (Matrix)", self.gpu_fp8_matrix, "TFLOPS", "Measures hardware-accelerated cooperative matrix quarter precision (8-bit) operations."),
                    metric_row("INT8", "INT8 (Vector)", self.gpu_int8_vector, "TOPS", "Measures vector 8-bit integer operations per second. Often used for quantized machine learning inference and specialized hardware-accelerated video processing."),
                    metric_row("INT8", "INT8 (Matrix)", self.gpu_int8_matrix, "TOPS", "Measures hardware-accelerated cooperative matrix 8-bit integer operations."),
                    metric_row("INT4", "INT4 (Vector)", self.gpu_int4_vector, "TOPS", "Measures vector 4-bit integer operations per second. An extreme quantization format used in ultra-efficient AI processing and specialized lookup tasks."),
                    metric_row("INT4", "INT4 (Matrix)", self.gpu_int4_matrix, "TOPS", "Measures hardware-accelerated cooperative matrix 4-bit integer operations."),
                ].spacing(12).into();

                let rt_content = column![
                    metric_row("RayTracing", "Intersect", self.gpu_rt_intersect, "GIS/s", "Measures raw intersection throughput against opaque triangle geometry. Tests the peak performance of the hardware's dedicated ray intersection engines."),
                    metric_row("RayDivergence", "Divergence", self.gpu_rt_divergence, "GRays/s", "Evaluates performance when neighboring rays hit entirely different materials and geometry, causing execution divergence and stalling compute wave-fronts."),
                    metric_row("RayAnyHit", "AnyHit", self.gpu_rt_anyhit, "GRays/s", "Tests intersection performance against geometry with alpha-testing (transparency) enabled. Stresses the GPU's ability to evaluate shaders during ray traversal."),
                    metric_row("RayIncoherent", "Incoherent", self.gpu_rt_incoherent, "GRays/s", "Tests performance when rays bounce in completely random directions, causing high cache misses. Simulates complex global illumination and path tracing."),
                    metric_row("RayPayload", "Payload", self.gpu_rt_payload, "GRays/s", "Tests the impact of carrying large blocks of data (payloads) along with the ray, which stresses the register usage and VRAM bandwidth of the compute units."),
                    metric_row("RayASBuild", "AS Build", self.gpu_rt_as_build, "ms", "Measures the time required to build the Acceleration Structure (BVH) for a complex scene. Lower is better. Crucial for dynamic or destructible environments."),
                    metric_row("RayProcedural", "Procedural", self.gpu_rt_procedural, "GRays/s", "Measures intersection speed against mathematically defined geometry (like spheres or curves) rather than explicit triangles. Useful for advanced rendering engines."),
                ].spacing(12).into();

                let compute_col = column![
                    text("COMPUTE CORES").size(16).style(color!(0xFFFFFF)),
                    Space::with_height(10),
                    create_panel("", color!(0xFF3366), compute_content)
                ].spacing(5).width(Length::FillPortion(1));

                let rt_col = column![
                    text("RAY TRACING").size(16).style(color!(0xFFFFFF)),
                    Space::with_height(10),
                    create_panel("", color!(0x00FF88), rt_content)
                ].spacing(5).width(Length::FillPortion(1));
                
                let mem_col = column![
                    text("MEMORY & SYSTEM").size(16).style(color!(0xFFFFFF)),
                    Space::with_height(10),
                    create_panel("", color!(0x00E5FF), sys_content)
                ].spacing(5).width(Length::FillPortion(1));

                let split_layout = row![compute_col, rt_col, mem_col].spacing(20);

                let save_btn: Element<'_, Message> = if matches!(self.state, AppState::Complete { .. }) {
                    button(
                        container(text("SAVE RESULTS").size(14).style(iced::theme::Text::Color(color!(0xFFFFFF))))
                            .width(Length::Fill)
                            .center_x()
                    )
                    .width(Length::Fill)
                    .padding([14, 0])
                    .on_press(Message::SaveResults)
                    .style(iced::theme::Button::Custom(Box::new(PrimaryGradientButton)))
                    .into()
                } else {
                    Space::with_height(0).into()
                };

                let sidebar = container(
                    column![
                        text("BENCHMARK").size(28).style(color!(0xFFFFFF)),
                        text("X PRO").size(28).style(color!(0x00E5FF)),
                        Space::with_height(50),
                        text("TARGET HARDWARE").size(11).style(color!(0x8888AA)),
                        Space::with_height(10),
                        text(if self.current_device.is_empty() { "Unknown" } else { &self.current_device }).size(13).style(color!(0xFFFFFF)),
                        Space::with_height(40),
                        text("STATUS").size(11).style(color!(0x8888AA)),
                        Space::with_height(10),
                        text(status_text).size(22).style(status_color),
                        Space::with_height(20),
                        text(if self.current_benchmark.len() > 30 { "Complete" } else { &self.current_benchmark }).size(12).style(color!(0x666677)),
                        Space::with_height(Length::Fill),
                        save_btn
                    ]
                )
                .width(Length::Fixed(300.0))
                .height(Length::Fill)
                .padding(30)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0A0F))),
                    border: Border { color: color!(0x1A1A24), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let main_area = container(
                    scrollable(
                        column![
                            global_progress,
                            split_layout
                        ].spacing(30)
                    ).height(Length::Fill)
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(40)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x111116))),
                    ..Default::default()
                });

                row![sidebar, main_area].width(Length::Fill).height(Length::Fill).into()
            }
        }
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}

impl GPUBenchApp {
    fn process_result(&mut self, res: &ResultData) {
        self.current_benchmark = res.benchmarkName.clone();
        if res.time_ms <= 0.0 { return; }
        
        let mut value = ((res.operations as f64) / (res.time_ms / 1000.0)) as f32;

        if res.metric == "ms/op" {
            value = (res.time_ms as f32) / (res.operations as f32);
        } else if res.metric == "ns" {
            value = ((res.time_ms * 1_000_000.0) as f32) / (res.operations as f32);
        } else if res.metric == "GIS/s" || res.metric == "GRays/s" || res.metric == "GB/s" {
            value /= 1e9;
        } else if res.metric == "TFLOPS" || res.metric == "TOPS" {
            value /= 1e12;
        }

        if res.backendName == "Native" || res.backendName == "System" {
            if res.component == "System" {
                if res.subcategory == "Bandwidth (Multi-threaded)" {
                    self.sys_mem_bw = self.sys_mem_bw.max(value);
                } else if res.subcategory == "Bandwidth (Single-threaded)" {
                    self.sys_mem_bw_single = self.sys_mem_bw_single.max(value);
                } else if res.subcategory == "Latency" {
                    if self.sys_mem_lat == 0.0 { self.sys_mem_lat = value; }
                    else { self.sys_mem_lat = self.sys_mem_lat.min(value); }
                }
            } else {
                self.cpu_bw = self.cpu_bw.max(value);
            }
        } else {
            match res.component.as_str() {
                "Memory" => {
                    self.gpu_bw = self.gpu_bw.max(value);
                }
                "Compute" => {
                    if res.subcategory == "FP64" { self.gpu_fp64 = self.gpu_fp64.max(value); }
                    if res.subcategory == "FP32" { self.gpu_fp32 = self.gpu_fp32.max(value); }
                    if res.subcategory == "FP16" {
                        if res.configIndex == 0 { self.gpu_fp16_vector = self.gpu_fp16_vector.max(value); }
                        else { self.gpu_fp16_matrix = self.gpu_fp16_matrix.max(value); }
                    }
                    if res.subcategory == "BF16" {
                        if res.configIndex == 0 { self.gpu_bf16_vector = self.gpu_bf16_vector.max(value); }
                        else { self.gpu_bf16_matrix = self.gpu_bf16_matrix.max(value); }
                    }
                    if res.subcategory == "FP8" {
                        if res.configIndex == 0 { self.gpu_fp8_vector = self.gpu_fp8_vector.max(value); }
                        else { self.gpu_fp8_matrix = self.gpu_fp8_matrix.max(value); }
                    }
                    if res.subcategory == "INT8" {
                        if res.configIndex == 0 { self.gpu_int8_vector = self.gpu_int8_vector.max(value); }
                        else { self.gpu_int8_matrix = self.gpu_int8_matrix.max(value); }
                    }
                    if res.subcategory == "INT4" {
                        if res.configIndex == 0 { self.gpu_int4_vector = self.gpu_int4_vector.max(value); }
                        else { self.gpu_int4_matrix = self.gpu_int4_matrix.max(value); }
                    }
                }
                "Ray Tracing" => {
                    if res.subcategory == "Alpha-Tested Geometry" { self.gpu_rt_anyhit = self.gpu_rt_anyhit.max(value); }
                    if res.subcategory == "AS Build Performance" {
                        if self.gpu_rt_as_build == 0.0 { self.gpu_rt_as_build = value; }
                        else { self.gpu_rt_as_build = self.gpu_rt_as_build.min(value); }
                    }
                    if res.subcategory == "Incoherent Traversal" { self.gpu_rt_incoherent = self.gpu_rt_incoherent.max(value); }
                    if res.subcategory == "Intersection tests" { self.gpu_rt_intersect = self.gpu_rt_intersect.max(value); }
                    if res.subcategory == "Material Divergence" || res.subcategory == "Execution Divergence" { self.gpu_rt_divergence = self.gpu_rt_divergence.max(value); }
                    if res.subcategory == "Payload Register Pressure" { self.gpu_rt_payload = self.gpu_rt_payload.max(value); }
                    if res.subcategory == "Procedural Intersection" { self.gpu_rt_procedural = self.gpu_rt_procedural.max(value); }
                }
                _ => {}
            }
        }
    }
}

fn create_panel<'a>(title: &str, title_color: iced::Color, children: Element<'a, Message>) -> iced::widget::Container<'a, Message> {
    container(column![
        text(title).size(22).style(title_color),
        Space::with_height(10),
        children
    ])
    .width(Length::Fill)
    .padding(20)
    .style(move |_t: &Theme| container::Appearance {
        background: Some(Background::Color(color!(0x111116))),
        border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x222233) },
        ..Default::default()
    })
}
