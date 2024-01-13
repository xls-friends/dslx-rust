// [WIP] parser for the DSLX language. Currently a spooky scary skeleton, but improving.
// Full language defined here: https://google.github.io/xls/dslx_reference/.
// TODO: It'd sure be nice to not have to `eat_whitespace` after every scanning fn.
//  - For now, tho, we consume leading whitespace in parsing fns (and not trailing).
use nom::{
    branch::alt,
    bytes::streaming::{tag, take_till},
    character::streaming::{alpha1, alphanumeric1},
    combinator::recognize,
    multi::{many0_count, separated_list0},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};

// AST nodes. Will be moved to their own file.
// TODO: There's currently a lot of copying in the AST how it might be set up; switch
// to single owner and reference holding once this is more fleshed out.

// Represents a name of an entity, such as a type, variable, function, ...
#[derive(Debug, PartialEq)]
pub struct Identifier {
    name: String,
}

// A parameter to a function, e.g., `foo: MyType`.
#[derive(Debug, PartialEq)]
pub struct Param {
    name: Identifier,
    // Will be made a "TypeRef".
    param_type: Identifier,
}

// A function!
// fn foo(x:u32) -> u32 { ... }
#[derive(Debug, PartialEq)]
pub struct FunctionSignature {
    name: Identifier,
    params: Vec<Param>,
    // Will be made a "TypeRef".
    ret_type: Identifier,
}

// Consumes all whitespace at the head of `input` and return a reference to the remaining string.
pub fn eat_whitespace(input: &str) -> Result<&str, nom::Err<nom::error::Error<&str>>> {
    // Using a lambda instead of the nom::character functions, as they're
    // ASCII-specific.
    Ok(take_till(|c: char| !c.is_whitespace())(input)?.0)
}

// Returns a parser that consumes preceding whitespace then runs the given parser.
pub fn preceding_whitespace<'a, O, P>(
    mut parser: P,
) -> impl FnMut(&'a str) -> IResult<&str, O, nom::error::Error<&'a str>>
where
    P: nom::Parser<&'a str, O, nom::error::Error<&'a str>>,
{
    move |input: &str| {
        let remaining = eat_whitespace(input)?;
        parser.parse(remaining)
    }
}

// Returns a "tag" parser that removes any preceding whitespace.
pub fn tag_ws<'a>(
    to_match: &'a str,
) -> impl FnMut(&'a str) -> IResult<&'a str, &str, nom::error::Error<&'a str>> {
    preceding_whitespace(tag(to_match))
}

// Parses a valid DSLX identifier, currently [_A-Za-z][_A-Za-z0-9]*.
// TODO: Use something aside from `alpha1`, etc., to support non-ASCII.
pub fn parse_identifier(input: &str) -> IResult<&str, Identifier> {
    let p = recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ));
    let mut ws_p = preceding_whitespace(p);
    ws_p(input).map(|(r, name)| {
        (
            r,
            Identifier {
                name: name.to_string(),
            },
        )
    })
}

// Parses a single param, e.g., `x: u32`.
fn parse_param(input: &str) -> IResult<&str, Param> {
    let name = parse_identifier;
    let param_type = preceded(tag_ws(":"), parse_identifier);
    let mut p = tuple((name, param_type));
    p(input).map(|(remaining, (n, pt))| {
        (
            remaining,
            Param {
                name: n,
                param_type: pt,
            },
        )
    })
}

// Parses a comma-separated list of params, e.g., `x: u32, y: MyCustomType`.
// Note that the list must _not_ end with a comma.
fn parse_param_list0(input: &str) -> IResult<&str, Vec<Param>> {
    separated_list0(preceding_whitespace(tag(",")), parse_param)(input)
}

// Parses a function signature, e.g.:
// `fn foo(a: u32, b: u64) -> uN[128]`
fn parse_function_signature(input: &str) -> IResult<&str, FunctionSignature> {
    let name = preceded(tag_ws("fn"), parse_identifier);
    let parameters = delimited(tag_ws("("), parse_param_list0, tag_ws(")"));
    let ret_type = preceded(tag_ws("->"), parse_identifier);
    let mut p = tuple((name, parameters, ret_type));
    p(input).map(|(remaining, (n, p, r))| {
        (
            remaining,
            FunctionSignature {
                name: n,
                params: p,
                ret_type: r,
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decent first stopping spot: can we parse a function signature?
    // TODO: Parse the rest of the fn.
    #[test]
    fn parse_fn_signature() {
        let input = "fn add_1(x: u32) -> u32 { x + u32:1 }";
        let expected = FunctionSignature {
            name: Identifier {
                name: "add_1".to_string(),
            },
            params: vec![Param {
                name: Identifier {
                    name: "x".to_string(),
                },
                param_type: Identifier {
                    name: "u32".to_string(),
                },
            }],
            ret_type: Identifier {
                name: "u32".to_string(),
            },
        };
        assert_eq!(
            parse_function_signature(&input),
            Ok((" { x + u32:1 }", expected))
        )
    }
}
