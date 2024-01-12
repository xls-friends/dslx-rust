// [WIP] parser for the DSLX language. Currently a spooky scary skeleton, but improving.
// Full language defined here: https://google.github.io/xls/dslx_reference/.
// TODO: It'd sure be nice to not have to `eat_whitespace` after every scanning fn.
//  - For now, tho, we consume leading whitespace in parsing fns (and not trailing).
use nom::{
    branch::alt,
    bytes::streaming::{
        tag, take_till,
    },
    character::streaming::{
        alpha1, alphanumeric1
    },
    combinator::recognize,
    multi::{ many0_count, separated_list0 },
    sequence::pair,
    IResult,
};

// AST nodes. Will be moved to their own file.
// TODO: There's currently a lot of copying in the AST how it might be set up; switch 
// to single owner and reference holding once this is more fleshed out.

// Represents a name of an entity, such as a type, variable, function, ...
#[derive(Debug, PartialEq)]
pub struct Identifier {
    name: String
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
pub struct FunctionNode {
    name: Identifier,
    params: Vec<Param>,
    // Will be made a "TypeRef".
    ret_type: Identifier,
}

// Consumes all whitespace at the head of `input` and return a reference to the remaining string.
pub fn eat_whitespace(input: &str) -> Result<&str, nom::Err<nom::error::Error<&str>>> {
    // Using a lambda instead of the nom::character functions, as they're
    // ASCII-specific.
    Ok(take_till(|c: char| { !c.is_whitespace() })(input)?.0)
}

// Returns a parser that consumes preceding whitespace then runs the given parser.
pub fn preceding_whitespace<'a, O, P>(mut parser: P,)
	-> impl FnMut(&'a str) -> IResult<&str, O, nom::error::Error<&'a str>>
	where
    P: nom::Parser<&'a str, O, nom::error::Error<&'a str>> {
	move |input: &str| {
		let remaining = eat_whitespace(input)?;
		parser.parse(remaining)
	}
}

// Parses a valid DSLX identifier, currently [_A-Za-z][_A-Za-z0-9]*.
// TODO: Use something aside from `alpha1`, etc., to support non-ASCII.
pub fn parse_identifier(input: &str) -> IResult<&str, Identifier> {
    let remaining = eat_whitespace(input)?;
    let (remaining, id) = recognize(
        pair(
            alt((alpha1, tag("_"))),
            many0_count(alt((alphanumeric1, tag("_"))))
        )
    )(remaining)?;
    Ok((remaining, Identifier { name: id.to_string() }))
}

// Parses a single param, e.g., `x: u32`.
fn parse_param(input: &str) -> IResult<&str, Param> {
    let remaining = eat_whitespace(input)?;
    let (mut remaining, param_name) = parse_identifier(remaining)?;
    remaining = eat_whitespace(remaining)?;
    (remaining, _) = tag(":")(remaining)?;
    // TODO: This should be changed to `parse_typeref`...once TypeRefs exist.
    let (remaining, param_type) = parse_identifier(remaining)?;
    Ok((remaining, Param { name: param_name, param_type: param_type }))
}

// Parses a comma-separated list of params, e.g., `x: u32, y: MyCustomType`.
// Note that the list must _not_ end with a comma.
fn parse_param_list0(input: &str) -> IResult<&str, Vec<Param>> {
    separated_list0(preceding_whitespace(tag(",")), parse_param)(input)
}

// Parses a function, e.g.:
// ```
//   fn foo(a: u32, b: u64) -> uN[128] {
//      a as uN[128] + b as uN[128]
//   }
// ```
// TODO: Parse beyond the signature.
fn parse_function(input: &str) -> IResult<&str, FunctionNode> {
    let remaining = preceding_whitespace(tag("fn"))(input)?.0;
    let (mut remaining, id) = parse_identifier(remaining)?;
    remaining = preceding_whitespace(tag("("))(remaining)?.0;
	let (mut remaining, params) = parse_param_list0(remaining)?;
    remaining = preceding_whitespace(tag(")"))(remaining)?.0;
    remaining = preceding_whitespace(tag("->"))(remaining)?.0;
    let (remaining, ret_type) = parse_identifier(remaining)?;

    Ok((remaining, FunctionNode {
        name: id,
        params: params,
        ret_type: ret_type,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decent first stopping spot: can we parse a function signature?
    // TODO: Parse the rest of the fn.
    #[test]
    fn parse_fn_signature() {
        let input = "fn add_1(x: u32) -> u32 { x + u32:1 }";
        let expected = FunctionNode {
            name: Identifier { name: "add_1".to_string() },
            params: vec![
                Param {
                    name: Identifier { name: "x".to_string() },
                    param_type: Identifier { name: "u32".to_string() },
                }
            ],
            ret_type: Identifier { name: "u32".to_string() },
        };
        assert_eq!(parse_function(&input), Ok((" { x + u32:1 }", expected)))
    }
}
