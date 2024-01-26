use nom_locate::LocatedSpan;

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
}

impl From<(usize, usize, usize)> for Pos {
    fn from(item: (usize, usize, usize)) -> Self {
        Pos {
            input_offset: item.0,
            line: item.1,
            column: item.2,
        }
    }
}

/// Identifies a range of the parser input.
#[derive(Debug, PartialEq)]
pub struct Span {
    pub start: Pos,
    /// Not inclusive, i.e., this represents the first position after the end of the spanned region.
    // TODO: Make this inclusive (not completely trivial, since it might
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
}

impl From<((usize, usize, usize), (usize, usize, usize))> for Span {
    fn from(item: ((usize, usize, usize), (usize, usize, usize))) -> Self {
        Span {
            start: Pos::from(item.0),
            end: Pos::from(item.1),
        }
    }
}

/// Represents a name of an entity, such as a type, variable, function, etc.
///
/// "Raw" means "not spanned".
#[derive(Debug, PartialEq)]
pub struct RawIdentifier<'a> {
    pub name: &'a str,
}

impl<'a> From<LocatedSpan<&'a str>> for RawIdentifier<'a> {
    fn from(span: LocatedSpan<&'a str>) -> Self {
        RawIdentifier {
            name: span.fragment(),
        }
    }
}

/// A parameter to a function, e.g., `foo: MyType`.
#[derive(Debug, PartialEq)]
pub struct RawParameter<'a> {
    pub name: Identifier<'a>,
    pub param_type: Identifier<'a>,
}

impl<'a> From<(Identifier<'a>, Identifier<'a>)> for RawParameter<'a> {
    fn from((name, param_type): (Identifier<'a>, Identifier<'a>)) -> Self {
        RawParameter { name, param_type }
    }
}

/// A function signature, e.g: `fn foo(x:u32) -> u32`.
#[derive(Debug, PartialEq)]
pub struct RawFunctionSignature<'a> {
    pub name: Identifier<'a>,
    pub parameters: ParameterList<'a>,
    pub result_type: Identifier<'a>,
}

impl<'a> From<(Identifier<'a>, ParameterList<'a>, Identifier<'a>)> for RawFunctionSignature<'a> {
    fn from(
        (name, parameters, result_type): (Identifier<'a>, ParameterList<'a>, Identifier<'a>),
    ) -> Self {
        RawFunctionSignature {
            name,
            parameters,
            result_type,
        }
    }
}

/// A parsed thing (e.g. `Identifier`) and the corresponding Span in the source text.
#[derive(Debug, PartialEq)]
pub struct Spanned<Thing> {
    pub span: Span,
    pub thing: Thing,
}

pub type Identifier<'a> = Spanned<RawIdentifier<'a>>;
pub type Parameter<'a> = Spanned<RawParameter<'a>>;
pub type ParameterList<'a> = Spanned<Vec<Parameter<'a>>>;
pub type FunctionSignature<'a> = Spanned<RawFunctionSignature<'a>>;
