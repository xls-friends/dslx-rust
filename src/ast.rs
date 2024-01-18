// Defines the types in the AST for DSLX.
pub type ParseInput<'a> = nom_locate::LocatedSpan<&'a str>;

// TODO: Move to own file.
// A distinct position in the input: offset from input start, line number, and column within line.
#[derive(Debug, PartialEq)]
pub struct Pos {
    pub input_offset: usize,
    pub line: usize,
    pub column: usize,
}

impl Pos {
    pub fn from_parse_input(x: ParseInput) -> Pos {
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

// An inclusive range of the input.
#[derive(Debug, PartialEq)]
pub struct Span {
    pub start: Pos,
    pub end: Pos,
}

impl Span {
    pub fn from_parse_input(start: ParseInput, end: ParseInput) -> Span {
        Span {
            start: Pos::from_parse_input(start),
            end: Pos::from_parse_input(end),
        }
    }

    pub fn from_values(start: (usize, usize, usize), end: (usize, usize, usize)) -> Span {
        Span {
            start: Pos::from_values(start.0, start.1, start.2),
            end: Pos::from_values(end.0, end.1, end.2),
        }
    }
}

// Represents a name of an entity, such as a type, variable, function, ...
#[derive(Debug, PartialEq)]
pub struct Identifier<'a> {
    pub span: Span,
    pub name: &'a str,
}

// A parameter to a function, e.g., `foo: MyType`.
#[derive(Debug, PartialEq)]
pub struct Param<'a> {
    pub span: Span,
    pub name: Identifier<'a>,
    pub param_type: Identifier<'a>,
}

// A function signature, e.g: `fn foo(x:u32) -> u32`.
#[derive(Debug, PartialEq)]
pub struct FunctionSignature<'a> {
    pub span: Span,
    pub name: Identifier<'a>,
    pub params: Vec<Param<'a>>,
    pub ret_type: Identifier<'a>,
}
