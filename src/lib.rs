// [WIP] parser for the DSLX language. Currently a spooky scary skeleton, but improving.
// TODO: It'd sure be nice to not have to `eat_whitespace` after every scanning fn.
use nom::{
    branch::alt,
    bytes::streaming::{
        tag, take_till,
    },
    character::streaming::{
        alpha1, alphanumeric1
    },
    combinator::recognize,
    multi::many0_count,
    sequence::pair,
    IResult,
};

// AST nodes. (Will be moved to their own file).
#[derive(Debug, PartialEq)]
struct Param {
    name: String,
    // Will be made a "TypeRef".
    param_type: String,
}

#[derive(Debug, PartialEq)]
struct FunctionNode {
    name: String,
    params: Vec<Param>,
    // Will be made a "TypeRef".
    ret_type: String,
}

// Consumes all whitespace at the head of `input`.
pub fn eat_whitespace(input: &str) -> IResult<&str, &str> {
    // Using a lambda instead of the nom::character functions, as they're
    // ASCII-specific.
    take_till(|c: char| { !c.is_whitespace() })(input)
}

// Parses a valid DSLX identifier, currently [_A-Za-z][_A-Za-z0-9]*.
// TODO: Use something aside from `alpha1`, etc., to support non-ASCII.
pub fn parse_identifier(input: &str) -> IResult<&str, &str> {
    recognize(
        pair(
            alt((alpha1, tag("_"))),
            many0_count(alt((alphanumeric1, tag("_"))))
        )
    )(input)
}

// Parses a comma-separated list of 
fn parse_param_list(input: &str) -> IResult<&str, Vec<Param>> {
    let mut params = Vec::new();

    // This could maybe be combined into many0(delimited...), but it seems less clear that way.
    // /shrug.
    let mut left = input;
    let left = loop {
        (left, _) = eat_whitespace(left)?;
        let (new_left, param_name) = parse_identifier(left)?;
        left = new_left;
        (left, _) = eat_whitespace(left)?;
        (left, _) = tag(":")(left)?;
        (left, _) = eat_whitespace(left)?;
        let (new_left, param_type) = parse_identifier(left)?;
        left = new_left;
        params.push(Param { name: param_name.to_string(), param_type: param_type.to_string() });

        let foo = tag::<&str, &str, nom::error::Error<&str>>(",")(left);
        if foo.is_err() {
            break left;
        };
        left = foo.unwrap().0;
    };
    let (left, _) = tag(")")(left)?;
    Ok((left, params))
}

// Returns the AstNode for "fn".
// TODO: Parse beyond the signature.
fn parse_function(input: &str) -> IResult<&str, FunctionNode> {
    let (mut left, _) = tag("fn")(input)?;
    (left, _) = eat_whitespace(left)?;
    let (mut left, id) = parse_identifier(left)?;
    (left, _) = eat_whitespace(left)?;
    (left, _) = tag("(")(left)?;
    // Is there a cleaner way to do "this or that" aside from an error?
    // Is that nom::opt?
    let (mut left, params) = match tag::<&str, &str, nom::error::Error<&str>>(")")(left) {
        Ok(res) => (res.0, Vec::new()),
        Err(_res) => { parse_param_list(left)? }
    };
    (left, _) = eat_whitespace(left)?;
    (left, _) = tag("->")(left)?;
    (left, _) = eat_whitespace(left)?;
    let (mut left, ret_type) = parse_identifier(left)?;
    (left, _) = eat_whitespace(left)?;

    Ok((left, FunctionNode {
        name: id.to_string(),
        params: params,
        ret_type: ret_type.to_string(),
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
            name: "add_1".to_string(),
            params: vec![
                Param {
                    name: "x".to_string(),
                    param_type: "u32".to_string(),
                }
            ],
            ret_type: "u32".to_string()
        };
        assert_eq!(parse_function(&input), Ok(("{ x + u32:1 }", expected)))
    }
}
