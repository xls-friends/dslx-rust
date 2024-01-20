// Defines the types in the AST for DSLX.
pub type ParseInput<'a> = nom_locate::LocatedSpan<&'a str>;

// TODO: Move pos/span to own file.
/// A distinct position in the input: offset from input start, line number, and column within line.
#[derive(Debug, PartialEq)]
pub struct Pos {
    /// Offset from the start of overall parse input in bytes/characters.
    /// Unicode is not _currently_ supported.
    pub input_offset: usize,
    /// 1-indexed line number from parse input.
    pub line: usize,
    /// 1-indexed column number in the current line.
    pub column: usize,
}

impl Pos {
    pub fn new(x: ParseInput) -> Pos {
        Pos {
            input_offset: x.location_offset(),
            line: x.location_line() as usize,
            column: x.get_column(),
        }
    }

    pub fn from_values(offset: usize, line: usize, column: usize) -> Pos {
        Pos {
            input_offset: offset,
            line: line,
            column: column,
        }
    }
}

/// An inclusive range of the input.
#[derive(Debug, PartialEq)]
pub struct Span {
    pub start: Pos,
    /// Not inclusive, i.e., this represents the first position after the end of the spanned region.
    // TODO: This is inelegant; make this inclusive (not completely trivial, since it might
    // require "backing up" a line & newline handling, ... Not horribly hard, either, just adequate
    // for its own change).
    pub end: Pos,
}

impl Span {
    pub fn new(start: ParseInput, end: ParseInput) -> Span {
        Span {
            start: Pos::new(start),
            end: Pos::new(end),
        }
    }

    pub fn from_values(start: (usize, usize, usize), end: (usize, usize, usize)) -> Span {
        Span {
            start: Pos::from_values(start.0, start.1, start.2),
            end: Pos::from_values(end.0, end.1, end.2),
        }
    }
}

/// Represents a name of an entity, such as a type, variable, function, ...
#[derive(Debug, PartialEq)]
pub struct Identifier<'a> {
    pub span: Span,
    pub name: &'a str,
}

/// A parameter to a function, e.g., `foo: MyType`.
#[derive(Debug, PartialEq)]
pub struct Param<'a> {
    pub span: Span,
    pub name: Identifier<'a>,
    pub param_type: Identifier<'a>,
}

/// A function signature, e.g: `fn foo(x:u32) -> u32`.
#[derive(Debug, PartialEq)]
pub struct FunctionSignature<'a> {
    pub span: Span,
    pub name: Identifier<'a>,
    pub params: Vec<Param<'a>>,
    pub ret_type: Identifier<'a>,
}
