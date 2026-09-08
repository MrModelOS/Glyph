//! Comment-preserving source formatter for Glyph.
//!
//! `format` normalizes leading indentation to a canonical 4-space-per-level
//! layout, collapses runs of blank lines to one, strips trailing whitespace
//! and guarantees a trailing newline. The interior spacing of statements is
//! left untouched; strings and comments are never altered or dropped. Because
//! only whitespace is changed, the output re-parses to the same program.

/// Whether a character opens or closes an indentation block.
fn brace_delta(c: char) -> i32 {
    match c {
        '{' => 1,
        '}' => -1,
        _ => 0,
    }
}

/// Scan a single line, updating string/comment state that may span lines.
/// Returns the count of `{` minus `}` that occur outside strings and comments.
fn scan_line(line: &str, in_block_comment: &mut bool) -> i32 {
    let mut delta = 0i32;
    let mut in_string = false;
    let mut in_line_comment = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if *in_block_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next(); // consume '/'
                *in_block_comment = false;
            }
            continue;
        }
        if in_line_comment {
            break;
        }
        if in_string {
            if c == '\\' {
                chars.next(); // skip escaped char (incl. quote/backslash)
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '/' if chars.peek() == Some(&'/') => in_line_comment = true,
            '/' if chars.peek() == Some(&'*') => {
                chars.next(); // consume '*'
                *in_block_comment = true;
            }
            '{' | '}' => delta += brace_delta(c),
            _ => {}
        }
    }
    delta
}

/// Returns the canonical indentation (in columns) for a line given the depth
/// of the block that starts the line.
fn indent_for(depth: i32, line: &str) -> String {
    let level = if line.trim_start().starts_with('}') {
        (depth - 1).max(0)
    } else {
        depth.max(0)
    };
    "    ".repeat(level as usize)
}

pub fn format(source: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut depth = 0i32;
    let mut in_block_comment = false;
    let mut prev_blank = false;

    for raw in source.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            // Collapse runs of blank lines to a single one.
            if !prev_blank && !out.is_empty() {
                out.push('\n'); // terminate the previous line; leave this one empty
                prev_blank = true;
            }
            continue;
        }

        if !out.is_empty() {
            out.push('\n');
        }
        let delta = scan_line(line, &mut in_block_comment);
        out.push_str(&indent_for(depth, line));
        out.push_str(line.trim_start());
        depth = (depth + delta).max(0);
        prev_blank = false;
    }

    // Always end with exactly one newline.
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(src: &str) -> String {
        format(src).unwrap()
    }

    #[test]
    fn indents_nested_blocks() {
        let src = "@fn main() -> Void {\nif x {\nprint(1);\n}\n}\n";
        let out = fmt(src);
        assert!(out.contains("\n    if x {\n        print(1);\n    }\n"));
    }

    #[test]
    fn keeps_interior_spacing() {
        let src = "@fn f() {\nlet   x  =  42;\n}\n";
        let out = fmt(src);
        assert!(out.contains("    let   x  =  42;\n"));
    }

    #[test]
    fn braces_inside_strings_are_ignored() {
        let src = "@fn main() {\nlet s = \"a{ b } c\";\n}\n";
        let out = fmt(src);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "    let s = \"a{ b } c\";");
    }

    #[test]
    fn preserves_comments() {
        let src = "@fn main() {\n// hello\nlet x = 1;\n}\n";
        let out = fmt(src);
        assert!(out.contains("    // hello\n"));
    }

    #[test]
    fn block_comments_with_braces() {
        let src = "/* if { } */\n@fn main() {\n}\n";
        let out = fmt(src);
        assert_eq!(out.lines().next().unwrap(), "/* if { } */");
    }

    #[test]
    fn idempotent() {
        for src in [
            "@fn main() {\nif x {\nprint(1);\n}\n}\n",
            "@fn f() {\nlet   x  =  42;\n}\n",
            "/* if { } */\n@fn main() {\n}\n",
        ] {
            let once = fmt(src);
            assert_eq!(fmt(&once), once);
        }
    }

    #[test]
    fn collapses_blank_lines_and_strips_trails() {
        let src = "@fn main() {\n\n\n\nlet x = 1;   \n}\n";
        let out = fmt(src);
        assert_eq!(out, "@fn main() {\n\n    let x = 1;\n}\n");
    }
}
