//! [WIP] parser for the DSLX language.
//!
//! Full language defined here: https://google.github.io/xls/dslx_reference/.
//! Currently a spooky scary skeleton, but actively being built out.
//!
//! At present, the [only] entry point is `parse_function_signature`, taking in an ast::ParseInput.
use nom::{
    branch::alt,
    bytes::streaming::{tag, take_till},
    character::streaming::{alpha1, alphanumeric1},
    combinator::recognize,
    multi::{many0_count, separated_list0},
    sequence::{delimited, pair, preceded, tuple},
    IResult, Parser,
};

pub mod ast;

use ast::{FunctionSignature, Identifier, ParameterList, ParseInput, RawParameter, Span, Spanned};

/// Return type for most parsing functions: takes in ParseInput and returns the `O` type or error.
type ParseResult<'a, O> = IResult<ParseInput<'a>, O, nom::error::Error<ParseInput<'a>>>;

/// Returns a parser that consumes preceding whitespace then runs the given parser.
pub fn preceding_whitespace<'a, O, P>(parser: P) -> impl FnMut(ParseInput<'a>) -> ParseResult<O>
where
    P: nom::Parser<ParseInput<'a>, O, nom::error::Error<ParseInput<'a>>>,
{
    preceded(take_till(|c: char| !c.is_whitespace()), parser)
}

/// Returns a "tag" parser that removes any preceding whitespace.
pub fn tag_ws<'a>(to_match: &'a str) -> impl FnMut(ParseInput<'a>) -> ParseResult<ParseInput<'a>> {
    preceding_whitespace(tag(to_match))
}

/// Returns the current position after consuming present whitespace.
pub fn position_ws<'a>() -> impl FnMut(ParseInput<'a>) -> ParseResult<ParseInput<'a>> {
    preceding_whitespace(nom_locate::position)
}

/// Returns a parser that captures the span encompassing the entirety of the given parser's
/// matched region, ignoring any preceding whitespace. Also converts (using
/// `Spanned::<Final>::from`) from the parser's intermediate result (`Intermediate`) to the
/// desired result type (`Final`).
///
/// `Intermediate`: the type produced by `parser`. E.g. for Identifier, this is a LocatedSpan.
///
/// `Final`: the type held inside the returned `Spanned`.
pub fn spanned<'a, Intermediate, Final, P>(
    parser: P,
) -> impl FnMut(ParseInput<'a>) -> ParseResult<Spanned<Final>>
where
    P: nom::Parser<ParseInput<'a>, Intermediate, nom::error::Error<ParseInput<'a>>>,
    Final: From<Intermediate>,
{
    nom::combinator::map(
        tuple((position_ws(), parser, nom_locate::position)),
        |(start, parser_result, end)| Spanned {
            span: Span::new(start, end),
            thing: Final::from(parser_result),
        },
    )
}

/// Parses a valid DSLX identifier, currently [_A-Za-z][_A-Za-z0-9]*.
///
/// # Example identifier
/// _Foobar123
pub fn parse_identifier(input: ParseInput) -> ParseResult<Identifier> {
    let p = recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ));
    spanned(p).parse(input)
}

/// Parses a single param, e.g., `x: u32`.
fn parse_param(input: ParseInput) -> ParseResult<Spanned<RawParameter>> {
    spanned(tuple((
        parse_identifier,
        preceded(tag_ws(":"), parse_identifier),
    )))
    .parse(input)
}

/// Parses a comma-separated list of params, e.g., `x: u32, y: MyCustomType`.
/// Note that a trailing comma will not be matched or consumed by this function.
fn parse_param_list0(input: ParseInput) -> ParseResult<ParameterList> {
    spanned(separated_list0(tag_ws(","), parse_param))(input)
}

/// Parses a function signature, e.g.:
/// `fn foo(a: u32, b: u64) -> uN[128]`
fn parse_function_signature(input: ParseInput) -> ParseResult<FunctionSignature> {
    let name = preceded(tag_ws("fn"), parse_identifier);
    let parameters = delimited(tag_ws("("), parse_param_list0, tag_ws(")"));
    let ret_type = preceded(tag_ws("->"), parse_identifier);
    spanned(tuple((name, parameters, ret_type))).parse(input)
}

#[cfg(test)]
mod tests {
    use crate::ast::{Parameter, RawFunctionSignature, RawIdentifier, RawParameter};

    use super::*;

    // TODO: Parse the rest of the fn.
    #[test]
    fn parse_fn_signature() -> Result<(), String> {
        // FIXME this fails if I delete the trailing space (i.e. u16" fails).
        let input = ParseInput::new("fn add_1(x: u32) -> u16 ");
        let expected = FunctionSignature {
            span: Span::from(((0, 1, 1), (23, 1, 24))),
            thing: RawFunctionSignature {
                name: Identifier {
                    span: Span::from(((3, 1, 4), (8, 1, 9))),
                    thing: RawIdentifier { name: "add_1" },
                },
                parameters: ParameterList {
                    span: Span::from(((9, 1, 10), (15, 1, 16))),
                    thing: vec![Parameter {
                        span: Span::from(((9, 1, 10), (15, 1, 16))),
                        thing: RawParameter {
                            name: Identifier {
                                span: Span::from(((9, 1, 10), (10, 1, 11))),
                                thing: RawIdentifier { name: "x" },
                            },
                            param_type: Identifier {
                                span: Span::from(((12, 1, 13), (15, 1, 16))),
                                thing: RawIdentifier { name: "u32" },
                            },
                        },
                    }],
                },
                result_type: Identifier {
                    span: Span::from(((20, 1, 21), (23, 1, 24))),
                    thing: RawIdentifier { name: "u16" },
                },
            },
        };
        let parsed = match parse_function_signature(input) {
            Ok(foo) => foo.1,
            Err(bar) => return Err(bar.to_string()),
        };
        assert_eq!(parsed, expected);
        Ok(())
    }
}
