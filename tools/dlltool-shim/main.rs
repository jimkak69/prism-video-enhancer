//! `dlltool` stand-in for the `x86_64-pc-windows-gnu` Rust toolchain.
//!
//! Crates built on `windows-link` declare `raw-dylib` imports. On the GNU toolchain rustc turns
//! those into import libraries by running GNU `dlltool`, which needs the GNU assembler (`as.exe`).
//! rustup's self-contained MinGW bundle ships `dlltool.exe` but no assembler, so builds fail with
//! "dlltool could not create import library".
//!
//! This program writes the same GNU-style import library that binutils dlltool produces, without
//! an assembler: a head object holding the import descriptor (`.idata$2`), one object per export
//! (jump thunk, `.idata$4/5` lookup and address entries, `.idata$6` hint/name) and a tail object
//! (terminators and the DLL name in `.idata$7`). The linker sorts `.idata$N` sections by object
//! name, so every library's entries stay contiguous behind its own descriptor even when several
//! crates import from the same DLL. (MSVC-style short import libraries do not have this property
//! with GNU ld and leave some import slots unbound.)
//!
//! rustc treats any stderr output from dlltool as failure, so this program writes to stderr only
//! when it fails. Set DLLTOOL_SHIM_LOG to a file path to log invocations.

use std::path::Path;
use std::process::ExitCode;

const MACHINE_AMD64: u16 = 0x8664;
const REL_ADDR32NB: u16 = 0x0003;
const REL_REL32: u16 = 0x0004;
const CLASS_EXTERNAL: u8 = 2;
const CLASS_STATIC: u8 = 3;
const SCN_TEXT: u32 = 0x6030_0020; // code | align 4 | execute | read
const SCN_IDATA4: u32 = 0xC030_0040; // initialized data | align 4 | read | write
const SCN_IDATA8: u32 = 0xC040_0040; // ... align 8
const SCN_IDATA2: u32 = 0xC020_0040; // ... align 2

struct Section {
    name: &'static str,
    flags: u32,
    data: Vec<u8>,
    relocs: Vec<(u32, usize, u16)>, // (offset, symbol index, type)
}

struct Symbol {
    name: String,
    value: u32,
    section: i16, // 1-based; 0 = undefined
    class: u8,
    is_function: bool,
}

#[derive(Default)]
struct Object {
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
}

impl Object {
    fn section(&mut self, name: &'static str, flags: u32, data: Vec<u8>) -> usize {
        self.sections.push(Section { name, flags, data, relocs: Vec::new() });
        let idx = self.sections.len();
        self.symbols.push(Symbol { name: name.to_string(), value: 0, section: idx as i16, class: CLASS_STATIC, is_function: false });
        idx
    }

    fn symbol(&mut self, name: &str, section: usize, class: u8, is_function: bool) -> usize {
        self.symbols.push(Symbol { name: name.to_string(), value: 0, section: section as i16, class, is_function });
        self.symbols.len() - 1
    }

    /// Index of the section symbol for a 1-based section number.
    fn section_symbol(&self, section: usize) -> usize {
        self.symbols.iter().position(|s| s.section == section as i16 && s.class == CLASS_STATIC).unwrap()
    }

    fn reloc(&mut self, section: usize, offset: u32, symbol: usize, kind: u16) {
        self.sections[section - 1].relocs.push((offset, symbol, kind));
    }

    fn write(&self) -> Vec<u8> {
        let mut strtab: Vec<u8> = Vec::new();
        let name_field = |name: &str, strtab: &mut Vec<u8>| -> [u8; 8] {
            let mut f = [0u8; 8];
            if name.len() <= 8 {
                f[..name.len()].copy_from_slice(name.as_bytes());
            } else {
                let off = 4 + strtab.len() as u32;
                strtab.extend_from_slice(name.as_bytes());
                strtab.push(0);
                f[4..].copy_from_slice(&off.to_le_bytes());
            }
            f
        };
        let header_len = 20 + 40 * self.sections.len();
        let mut body: Vec<u8> = Vec::new();
        let mut sec_headers: Vec<u8> = Vec::new();
        for s in &self.sections {
            let data_ptr = if s.data.is_empty() { 0 } else { (header_len + body.len()) as u32 };
            body.extend_from_slice(&s.data);
            if body.len() % 2 == 1 {
                body.push(0);
            }
            let reloc_ptr = if s.relocs.is_empty() { 0 } else { (header_len + body.len()) as u32 };
            for (off, sym, kind) in &s.relocs {
                body.extend_from_slice(&off.to_le_bytes());
                body.extend_from_slice(&(*sym as u32).to_le_bytes());
                body.extend_from_slice(&kind.to_le_bytes());
            }
            let mut h = Vec::with_capacity(40);
            h.extend_from_slice(&name_field(s.name, &mut strtab));
            h.extend_from_slice(&0u32.to_le_bytes()); // VirtualSize
            h.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
            h.extend_from_slice(&(s.data.len() as u32).to_le_bytes());
            h.extend_from_slice(&data_ptr.to_le_bytes());
            h.extend_from_slice(&reloc_ptr.to_le_bytes());
            h.extend_from_slice(&0u32.to_le_bytes()); // line numbers
            h.extend_from_slice(&(s.relocs.len() as u16).to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes());
            h.extend_from_slice(&s.flags.to_le_bytes());
            sec_headers.extend_from_slice(&h);
        }
        let symtab_ptr = (header_len + body.len()) as u32;
        let mut symtab: Vec<u8> = Vec::new();
        for s in &self.symbols {
            symtab.extend_from_slice(&name_field(&s.name, &mut strtab));
            symtab.extend_from_slice(&s.value.to_le_bytes());
            symtab.extend_from_slice(&s.section.to_le_bytes());
            symtab.extend_from_slice(&(if s.is_function { 0x20u16 } else { 0 }).to_le_bytes());
            symtab.push(s.class);
            symtab.push(0);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MACHINE_AMD64.to_le_bytes());
        out.extend_from_slice(&(self.sections.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&symtab_ptr.to_le_bytes());
        out.extend_from_slice(&(self.symbols.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&sec_headers);
        out.extend_from_slice(&body);
        out.extend_from_slice(&symtab);
        out.extend_from_slice(&((strtab.len() + 4) as u32).to_le_bytes());
        out.extend_from_slice(&strtab);
        out
    }

    fn defined_externals(&self) -> Vec<String> {
        self.symbols.iter().filter(|s| s.class == CLASS_EXTERNAL && s.section > 0).map(|s| s.name.clone()).collect()
    }
}

enum Export {
    Named { name: String, data: bool },
    Ordinal { name: String, ordinal: u16 },
}

fn parse_def(text: &str) -> Vec<Export> {
    let mut exports = Vec::new();
    let mut in_exports = false;
    for line in text.lines() {
        let line = line.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if upper.starts_with("LIBRARY") || upper.starts_with("NAME") {
            continue;
        }
        if upper == "EXPORTS" {
            in_exports = true;
            continue;
        }
        if !in_exports {
            continue;
        }
        let mut parts = line.split_whitespace();
        let mut name = parts.next().unwrap_or("").trim_matches('"').to_string();
        if let Some((n, _)) = name.split_once('=') {
            name = n.to_string();
        }
        let rest: Vec<&str> = parts.collect();
        let ordinal = rest.iter().find_map(|p| p.strip_prefix('@')).and_then(|o| o.parse::<u16>().ok());
        let noname = rest.iter().any(|p| p.eq_ignore_ascii_case("NONAME"));
        let data = rest.iter().any(|p| p.eq_ignore_ascii_case("DATA"));
        match (ordinal, noname) {
            (Some(o), true) => exports.push(Export::Ordinal { name, ordinal: o }),
            _ => exports.push(Export::Named { name, data }),
        }
    }
    exports
}

/// GNU `ar` archive with a symbol table and long-name table.
fn write_archive(members: &[(String, Vec<u8>, Vec<String>)]) -> Vec<u8> {
    let header = |name: &str, size: usize| -> Vec<u8> {
        let h = format!("{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n", name, 0, 0, 0, 644, size);
        assert_eq!(h.len(), 60);
        h.into_bytes()
    };
    let mut longnames = Vec::new();
    let mut names = Vec::new();
    for (n, _, _) in members {
        names.push(format!("/{}", longnames.len()));
        longnames.extend_from_slice(n.as_bytes());
        longnames.extend_from_slice(b"/\n");
    }
    let sym_count: usize = members.iter().map(|m| m.2.len()).sum();
    let sym_strings: usize = members.iter().flat_map(|m| m.2.iter()).map(|s| s.len() + 1).sum();
    let symtab_size = 4 + 4 * sym_count + sym_strings;
    let pad = |n: usize| n + (n & 1);
    let mut offset = 8 + 60 + pad(symtab_size) + 60 + pad(longnames.len());
    let mut offsets = Vec::new();
    for (_, data, _) in members {
        offsets.push(offset);
        offset += 60 + pad(data.len());
    }
    let mut out = b"!<arch>\n".to_vec();
    out.extend(header("/", symtab_size));
    out.extend_from_slice(&(sym_count as u32).to_be_bytes());
    for (i, m) in members.iter().enumerate() {
        for _ in &m.2 {
            out.extend_from_slice(&(offsets[i] as u32).to_be_bytes());
        }
    }
    for m in members {
        for s in &m.2 {
            out.extend_from_slice(s.as_bytes());
            out.push(0);
        }
    }
    if symtab_size % 2 == 1 {
        out.push(b'\n');
    }
    out.extend(header("//", longnames.len()));
    out.extend_from_slice(&longnames);
    if longnames.len() % 2 == 1 {
        out.push(b'\n');
    }
    for (i, (_, data, _)) in members.iter().enumerate() {
        out.extend(header(&names[i], data.len()));
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(b'\n');
        }
    }
    out
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

fn build(dll: &str, exports: &[Export], unique: &str) -> Vec<u8> {
    // Unique per library so several import libraries for one DLL never share a head.
    let tag: String = format!("{}_{:08x}", dll.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>(), fnv1a(unique) as u32);
    let head_sym = format!("_head_{tag}");
    let iname_sym = format!("__{tag}_iname");
    let mut members = Vec::new();

    // Head: import descriptor pointing at the (empty) start of this library's .idata$4/$5 runs.
    let mut head = Object::default();
    let s2 = head.section(".idata$2", SCN_IDATA4, vec![0u8; 20]);
    let s5 = head.section(".idata$5", SCN_IDATA8, Vec::new());
    let s4 = head.section(".idata$4", SCN_IDATA8, Vec::new());
    head.symbol(&head_sym, s2, CLASS_EXTERNAL, false);
    let iname = head.symbol(&iname_sym, 0, CLASS_EXTERNAL, false);
    let (sym4, sym5) = (head.section_symbol(s4), head.section_symbol(s5));
    head.reloc(s2, 0, sym4, REL_ADDR32NB);
    head.reloc(s2, 12, iname, REL_ADDR32NB);
    head.reloc(s2, 16, sym5, REL_ADDR32NB);
    members.push((format!("{tag}_h.o"), head.write(), head.defined_externals()));

    for (i, e) in exports.iter().enumerate() {
        let mut o = Object::default();
        let (name, ordinal, data) = match e {
            Export::Named { name, data } => (name.as_str(), None, *data),
            Export::Ordinal { name, ordinal } => (name.as_str(), Some(*ordinal), false),
        };
        let text = if data { None } else { Some(o.section(".text", SCN_TEXT, vec![0xFF, 0x25, 0, 0, 0, 0, 0x90, 0x90])) };
        let s7 = o.section(".idata$7", SCN_IDATA4, vec![0u8; 4]);
        let entry = match ordinal {
            Some(n) => (0x8000_0000_0000_0000u64 | n as u64).to_le_bytes().to_vec(),
            None => vec![0u8; 8],
        };
        let s5 = o.section(".idata$5", SCN_IDATA8, entry.clone());
        let s4 = o.section(".idata$4", SCN_IDATA8, entry);
        let s6 = if ordinal.is_none() {
            let mut hint = vec![0u8, 0u8];
            hint.extend_from_slice(name.as_bytes());
            hint.push(0);
            if hint.len() % 2 == 1 {
                hint.push(0);
            }
            Some(o.section(".idata$6", SCN_IDATA2, hint))
        } else {
            None
        };
        if let Some(t) = text {
            o.symbol(name, t, CLASS_EXTERNAL, true);
        }
        let imp = o.symbol(&format!("__imp_{name}"), s5, CLASS_EXTERNAL, false);
        let head_ref = o.symbol(&head_sym, 0, CLASS_EXTERNAL, false);
        if let Some(t) = text {
            o.reloc(t, 2, imp, REL_REL32);
        }
        o.reloc(s7, 0, head_ref, REL_ADDR32NB);
        if let Some(s6) = s6 {
            let hs = o.section_symbol(s6);
            o.reloc(s5, 0, hs, REL_ADDR32NB);
            o.reloc(s4, 0, hs, REL_ADDR32NB);
        }
        members.push((format!("{tag}_s{i:05}.o"), o.write(), o.defined_externals()));
    }

    // Tail: list terminators and the DLL name.
    let mut tail = Object::default();
    tail.section(".idata$4", SCN_IDATA8, vec![0u8; 8]);
    tail.section(".idata$5", SCN_IDATA8, vec![0u8; 8]);
    let mut name = dll.as_bytes().to_vec();
    name.push(0);
    if name.len() % 2 == 1 {
        name.push(0);
    }
    let s7 = tail.section(".idata$7", SCN_IDATA4, name);
    tail.symbol(&iname_sym, s7, CLASS_EXTERNAL, false);
    members.push((format!("{tag}_t.o"), tail.write(), tail.defined_externals()));

    write_archive(&members)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut def, mut out, mut dll, mut machine) = (None, None, None, String::from("i386:x86-64"));
    let mut i = 0;
    while i < args.len() {
        let next = args.get(i + 1).cloned();
        match args[i].as_str() {
            "-d" | "--input-def" => (def, i) = (next, i + 2),
            "-l" | "--output-lib" => (out, i) = (next, i + 2),
            "-D" | "--dllname" => (dll, i) = (next, i + 2),
            "-m" | "--machine" => {
                machine = next.unwrap_or_default();
                i += 2;
            }
            "-f" | "--as-flags" | "-S" | "--as" | "--temp-prefix" => i += 2,
            _ => i += 1,
        }
    }
    let (Some(def), Some(out)) = (def, out) else {
        eprintln!("dlltool-shim: expected -d <file.def> and -l <out.lib>, got {args:?}");
        return ExitCode::FAILURE;
    };
    if machine != "i386:x86-64" {
        eprintln!("dlltool-shim: only x86-64 is supported (got -m {machine})");
        return ExitCode::FAILURE;
    }
    let text = match std::fs::read_to_string(&def) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("dlltool-shim: cannot read {def}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dll = dll
        .or_else(|| text.lines().find_map(|l| l.trim().strip_prefix("LIBRARY").map(|s| s.trim().trim_matches('"').to_string())))
        .unwrap_or_else(|| Path::new(&out).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
    if let Some(log) = std::env::var_os("DLLTOOL_SHIM_LOG") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
            let _ = writeln!(f, "== {args:?}\n{text}");
        }
    }
    let exports = parse_def(&text);
    if let Err(e) = std::fs::write(&out, build(&dll, &exports, &out)) {
        eprintln!("dlltool-shim: cannot write {out}: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
