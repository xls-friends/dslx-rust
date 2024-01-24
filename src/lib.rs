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

use ast::ParseInput;

// Return type for most parsing functions: takes in ParseInput and returns the `O` type or error.
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

/// Gets the current position after consuming present whitespace.
pub fn position_ws<'a>() -> impl FnMut(ParseInput<'a>) -> ParseResult<ParseInput<'a>> {
    preceding_whitespace(nom_locate::position)
}

/// Returns a parser that captures the span encompassing the entirety of the specified parser's
/// matched region, ignoring any preceding whitespace.
pub fn spanned<'a, O, P>(parser: P) -> impl FnMut(ParseInput<'a>) -> ParseResult<(O, ast::Span)>
where
    P: nom::Parser<ParseInput<'a>, O, nom::error::Error<ParseInput<'a>>>,
{
    nom::combinator::map(
        tuple((position_ws(), parser, nom_locate::position)),
        |(start, stuff, end)| (stuff, ast::Span::new(start, end)),
    )
}

/// Parses a valid DSLX identifier, currently [_A-Za-z][_A-Za-z0-9]*.
pub fn parse_identifier(input: ParseInput) -> ParseResult<ast::Identifier> {
    let p = recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ));
    spanned(p)
        .map(|(name, span)| ast::Identifier {
            span: span,
            name: name.fragment(),
        })
        .parse(input)
}

/// Parses a single param, e.g., `x: u32`.
fn parse_param(input: ParseInput) -> ParseResult<ast::Param> {
    let p = spanned(tuple((
        parse_identifier,
        preceded(tag_ws(":"), parse_identifier),
    )));
    p.map(|((name, param_type), span)| ast::Param {
        span: span,
        name: name,
        param_type: param_type,
    })
    .parse(input)
}

/// Parses a comma-separated list of params, e.g., `x: u32, y: MyCustomType`.
/// Note that a trailing comma will not be matched or consumed by this function.
fn parse_param_list0(input: ParseInput) -> ParseResult<Vec<ast::Param>> {
    separated_list0(tag_ws(","), parse_param)(input)
}

/// Parses a function signature, e.g.:
/// `fn foo(a: u32, b: u64) -> uN[128]`
fn parse_function_signature(span: ParseInput) -> ParseResult<ast::FunctionSignature> {
    let name = preceded(tag_ws("fn"), parse_identifier);
    let parameters = delimited(tag_ws("("), parse_param_list0, tag_ws(")"));
    let ret_type = preceded(tag_ws("->"), parse_identifier);
    let p = spanned(tuple((name, parameters, ret_type)));
    p.map(|((n, p, r), span)| ast::FunctionSignature {
        span: span,
        name: n,
        params: p,
        ret_type: r,
    })
    .parse(span)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decent first stopping spot: can we parse a function signature?
    // TODO: Parse the rest of the fn.
    #[test]
    fn parse_fn_signature() -> Result<(), String> {
        let input = ParseInput::new("fn add_1(x: u32) -> u32 { x + u32:1 }");
        let expected = ast::FunctionSignature {
            span: ast::Span::from(((0, 1, 1), (23, 1, 24))),
            name: ast::Identifier {
                span: ast::Span::from(((3, 1, 4), (8, 1, 9))),
                name: "add_1",
            },
            params: vec![ast::Param {
                span: ast::Span::from(((9, 1, 10), (15, 1, 16))),
                name: ast::Identifier {
                    span: ast::Span::from(((9, 1, 10), (10, 1, 11))),
                    name: "x",
                },
                param_type: ast::Identifier {
                    span: ast::Span::from(((12, 1, 13), (15, 1, 16))),
                    name: "u32",
                },
            }],
            ret_type: ast::Identifier {
                span: ast::Span::from(((20, 1, 21), (23, 1, 24))),
                name: "u32",
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
