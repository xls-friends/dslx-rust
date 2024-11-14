//! **WIP** parser for the DSLX language.
//!
//! Full language defined here: <https://google.github.io/xls/dslx_reference/>
//! Currently a spooky scary skeleton, but actively being built out.
//!
//! At present, the _only_ entry point is `parse_function_signature`, taking in an ast::ParseInput.
#![feature(assert_matches)]
// TODO one day when all functions in this file are used, delete below. For now, we prefer to
// avoid spammy Github Actions notes about unused functions.
#![allow(dead_code)]
// We prerfer brevity.
#![allow(elided_named_lifetimes)]
pub mod ast;

use ast::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, char, digit1, hex_digit1, satisfy},
    combinator::{flat_map, map_opt, map_res, not, opt, peek, recognize, success, value, verify},
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult, Parser,
};
use nonempty::NonEmpty;
use num_bigint::{BigInt, BigUint};
use std::cmp::Ordering::Greater;

/// Return type for most parsing functions: takes in ParseInput and returns the `O` type or error.
type ParseResult<'a, O> = IResult<ParseInput<'a>, O, nom::error::Error<ParseInput<'a>>>;

/// Returns a parser that consumes preceding whitespace then runs the given parser.
pub fn preceding_whitespace<'a, O, P>(parser: P) -> impl FnMut(ParseInput<'a>) -> ParseResult<O>
where
    P: nom::Parser<ParseInput<'a>, O, nom::error::Error<ParseInput<'a>>>,
{
    preceded(take_while(|c: char| c.is_whitespace()), parser)
}

/// Returns a "tag" parser that removes any preceding whitespace.
pub fn tag_ws<'a>(to_match: &'a str) -> impl FnMut(ParseInput<'a>) -> ParseResult<ParseInput<'a>> {
    preceding_whitespace(tag(to_match))
}

/// Returns the current position after consuming preceding whitespace.
pub fn position_ws<'a>() -> impl FnMut(ParseInput<'a>) -> ParseResult<ParseInput<'a>> {
    preceding_whitespace(nom_locate::position)
}

/// A parser that consumes exactly 1 whitespace character.
pub fn whitespace_exactly1(input: ParseInput) -> ParseResult<()> {
    value((), satisfy(|c: char| c.is_whitespace())).parse(input)
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

/// Parses a valid DSLX identifier, currently \[_A-Za-z]\[_A-Za-z0-9]*.
///
/// # Example identifier
/// _Foobar123
pub fn parse_identifier(input: ParseInput) -> ParseResult<Identifier> {
    let p = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ));
    spanned(p).parse(input)
}

/// Parses a variable declaration, e.g., `x: u32`.
fn parse_variable_declaration(input: ParseInput) -> ParseResult<BindingDecl> {
    spanned(tuple((
        parse_identifier,
        preceded(tag_ws(":"), parse_identifier),
    )))
    .parse(input)
}

/// Parses a comma-separated list of variable declarations, e.g., `x: u32, y: MyCustomType`.
/// Note that a trailing comma will not be matched or consumed by this function.
fn parse_parameter_list0(input: ParseInput) -> ParseResult<BindingDeclList> {
    // TODO C++ DSLX allows a single trailing comma. Parse/allow that here.
    spanned(separated_list0(tag_ws(","), parse_variable_declaration))(input)
}

/// Parses a function signature, e.g.:
/// `fn foo(a: u32, b: u64) -> uN[128]`
fn parse_function_signature(input: ParseInput) -> ParseResult<FunctionSignature> {
    let name = preceded(tag_ws("fn"), parse_identifier);
    let parameters = delimited(tag_ws("("), parse_parameter_list0, tag_ws(")"));
    let ret_type = preceded(tag_ws("->"), parse_identifier);
    spanned(tuple((name, parameters, ret_type))).parse(input)
}

/// Parses an unsigned binary integer (including the `0b` prefix) of arbitrary length, e.g.:
///
/// `0b10`,
/// `0b0011`
///
/// At most one `_` is allowed between any two digits (to group digits), e.g.:
///
/// `0b1000_0001`
///
/// Does not consume preceding whitespace. The caller should do so.
fn parse_unsigned_binary(input: ParseInput) -> ParseResult<BigUint> {
    let prefix = tag("0b");
    let binary_digit1 = take_while1(|c: char| c == '0' || c == '1');
    let separated_digits = recognize(separated_list1(char('_'), binary_digit1));
    map_opt(preceded(prefix, separated_digits), |s: ParseInput| {
        BigUint::parse_bytes(s.fragment().as_bytes(), 2)
    })
    .parse(input)
}

/// Parses an unsigned decimal integer of arbitrary length, e.g.:
///
/// `0`,
/// `1`,
/// `1361129467683753853853498429727072845824`
///
/// At most one `_` is allowed between any two digits (to group digits), e.g.:
///
/// `100_000`
///
/// Does not consume preceding whitespace. The caller should do so.
fn parse_unsigned_decimal(input: ParseInput) -> ParseResult<BigUint> {
    let digits = recognize(separated_list1(char('_'), digit1));
    map_opt(digits, |s: ParseInput| {
        BigUint::parse_bytes(s.fragment().as_bytes(), 10)
    })
    .parse(input)
}

/// Parses an unsigned hexadecimal integer (including the `0x` prefix) of arbitrary length, e.g.:
///
/// `0xAbCd`,
/// `0x1f`,
/// `0xaB3`
///
/// At most one `_` is allowed between any two digits (to group digits), e.g.:
///
/// `0x10_01`
///
/// Does not consume preceding whitespace. The caller should do so.
fn parse_unsigned_hexadecimal(input: ParseInput) -> ParseResult<BigUint> {
    let prefix = tag("0x");
    let separated_digits = recognize(separated_list1(char('_'), hex_digit1));
    map_opt(preceded(prefix, separated_digits), |s: ParseInput| {
        BigUint::parse_bytes(s.fragment().as_bytes(), 16)
    })
    .parse(input)
}

/// Parses an unsigned decimal, hexadecimal, or binary integer.
///
/// Does not consume preceding whitespace. The caller should do so.
fn parse_unsigned_integer(input: ParseInput) -> ParseResult<BigUint> {
    let not_hex_or_binary_tag = not(peek(alt((tag("0b"), tag("0x")))));
    alt((
        // If the input starts with `0b` or `0x` we don't want parse_unsigned_decimal to consume
        // the 0 and stop parsing before the b or x.
        preceded(not_hex_or_binary_tag, parse_unsigned_decimal),
        parse_unsigned_hexadecimal,
        parse_unsigned_binary,
    ))
    .parse(input)
}

/// Parses a signed decimal, hexadecimal, or binary integer.
///
/// Does not consume preceding whitespace. The caller should do so.
fn parse_signed_integer(input: ParseInput) -> ParseResult<BigInt> {
    let negative = opt(char('-'));
    nom::combinator::map(
        tuple((negative, preceding_whitespace(parse_unsigned_integer))),
        |(neg, bu)| match neg {
            Some(_) => BigInt::from_biguint(num_bigint::Sign::Minus, bu),
            None => BigInt::from_biguint(num_bigint::Sign::Plus, bu),
        },
    )
    .parse(input)
}

/// Parses a bit type. E.g.:
///
/// `u3`
/// `uN[3]`
/// `bits[3]`
///
/// `s3`
/// `sN[3]`
///
/// See <https://google.github.io/xls/dslx_reference/#bit-type>
fn parse_bit_type(input: ParseInput) -> ParseResult<BitType> {
    // {s,u}
    let sign = alt((
        char('s').map(|_| Signedness::Signed),
        char('u').map(|_| Signedness::Unsigned),
    ));

    // A base10 integer between [0, 2^32)
    let decimal_to_2_32 = map_res(digit1, |s: ParseInput| s.fragment().parse::<u32>());
    let shorthand = verify(
        decimal_to_2_32,
        // "These are defined up to u64."
        |&width| width <= 64,
    );

    let decimal_to_2_32 = map_res(digit1, |s: ParseInput| s.fragment().parse::<u32>());

    // `N[1]`
    let n_brackets = delimited(
        tuple((char('N'), preceding_whitespace(char('[')))),
        preceding_whitespace(decimal_to_2_32),
        preceding_whitespace(char(']')),
    );

    let explicitly_signed_type = tuple((sign, alt((shorthand, n_brackets))));

    // `bits[1]`
    let decimal_to_2_32 = map_res(digit1, |s: ParseInput| s.fragment().parse::<u32>());
    let bits = nom::combinator::map(
        delimited(
            tuple((tag("bits"), preceding_whitespace(char('[')))),
            preceding_whitespace(decimal_to_2_32),
            preceding_whitespace(char(']')),
        ),
        |x| (Signedness::Unsigned, x),
    );

    spanned(alt((explicitly_signed_type, bits))).parse(input)
}

/// Parses a literal expression. E.g.,
///
/// `u8:12`, `u8:0b00001100`
///
/// `s8:128`, `s8:-128`
///
/// Uses the bit type to determine which kind of `RawInteger` to parse.
fn parse_literal<'a>(input: ParseInput<'a>) -> ParseResult<'a, Literal> {
    let parse = |input: ParseInput<'a>| -> ParseResult<'a, RawLiteral> {
        let (input, bit_type) = terminated(parse_bit_type, tag_ws(":"))(input)?;
        let (rest, value): (ParseInput, Integer) = match bit_type.thing.signedness {
            Signedness::Signed => spanned(parse_signed_integer).parse(input),
            Signedness::Unsigned => spanned(parse_unsigned_integer).parse(input),
        }?;
        Ok((rest, RawLiteral { value, bit_type }))
    };

    spanned(parse).parse(input)
}

/// Parses a unary operator. I.e. `!` and `-`
fn parse_unary_operator(input: ParseInput) -> ParseResult<UnaryOperator> {
    let op = alt((
        value(RawUnaryOperator::Negate, tag("-")),
        value(RawUnaryOperator::Invert, tag("!")),
    ));
    spanned(op).parse(input)
}

/// Parses a binary operator. E.g. `|`, `&`, etc.
fn parse_binary_operator(input: ParseInput) -> ParseResult<BinaryOperator> {
    let op = alt((
        // These two must match before | and &
        value(RawBinaryOperator::BooleanOr, tag("||")),
        value(RawBinaryOperator::BooleanAnd, tag("&&")),
        // now match | and &
        value(RawBinaryOperator::BitwiseOr, tag("|")),
        value(RawBinaryOperator::BitwiseAnd, tag("&")),
        value(RawBinaryOperator::BitwiseXor, tag("^")),
        // concatenate must match before +
        value(RawBinaryOperator::Concatenate, tag("++")),
        // the order of these does not matter
        // arithmetic
        value(RawBinaryOperator::Add, tag("+")),
        value(RawBinaryOperator::Subtract, tag("-")),
        value(RawBinaryOperator::Multiply, tag("*")),
        // shift
        value(RawBinaryOperator::ShiftRight, tag(">>")),
        value(RawBinaryOperator::ShiftLeft, tag("<<")),
        // compare
        value(RawBinaryOperator::Equal, tag("==")),
        value(RawBinaryOperator::NotEqual, tag("!=")),
        value(RawBinaryOperator::GreaterOrEqual, tag(">=")),
        value(RawBinaryOperator::Greater, tag(">")),
        value(RawBinaryOperator::LessOrEqual, tag("<=")),
        value(RawBinaryOperator::Less, tag("<")),
        // other
        value(RawBinaryOperator::Range, tag("..")),
    ));
    spanned(op).parse(input)
}

/// Parses a let expression. E.g. `let x : u32 = a + u32:1; x`
///
/// Note the trailing expression `x`. It is part of the let expression. It is common to have an
/// expression after the let (after all, what's the point of declaring a variable binding if
/// you're never going to use the variable?), but not required. So the trailing expression is
/// optional.
fn parse_let_expression<'a>(
    input: ParseInput<'a>,
) -> ParseResult<'a, (NonEmpty<LetBinding>, Option<Expression>)> {
    let parse_let_binding = |input: ParseInput<'a>| -> ParseResult<'a, RawLetBinding> {
        let var_decl = delimited(
            // let must be followed by at least 1 whitespace
            tuple((tag_ws("let"), whitespace_exactly1)),
            parse_variable_declaration,
            tag_ws("="),
        );
        let bound_expr = terminated(parse_expression(None), tag_ws(";"));
        tuple((var_decl, bound_expr))
            .map(|(variable_declaration, value)| RawLetBinding {
                variable_declaration,
                value: Box::new(value),
            })
            .parse(input)
    };

    // We avoid a recursive parsing implementation of nested let expressions to avoid stack
    // overflows when fuzzing. Furthermore, we want to be as robust as possible, and not make
    // assumptions about the user (i.e. not assume the user is going to limit their nesting of
    // lets).
    let bindings = many1(spanned(parse_let_binding)).map(|xs| {
        let mut ys = NonEmpty::new(xs.first().unwrap().clone());
        ys.extend(xs.into_iter().skip(1));
        ys
    });

    let using_expr = opt(parse_expression(None));
    tuple((bindings, using_expr)).parse(input)
}

/// Parses an `if...else if...else` expression. E.g.
/// `if condition0 { consequent0 } else if condition1 { consequent1 } else { alternate }`
fn parse_ifelse_expression<'a>(
    input: ParseInput<'a>,
) -> ParseResult<'a, (NonEmpty<ConditionConsequent>, Expression)> {
    let parse_if_condition_consequent =
        |input: ParseInput<'a>| -> ParseResult<'a, RawConditionConsequent> {
            let if_cond = preceded(
                // if must be followed by at least 1 whitespace
                tuple((tag_ws("if"), whitespace_exactly1)),
                parse_expression(None),
            );
            let consequent = delimited(tag_ws("{"), parse_expression(None), tag_ws("}"));
            tuple((if_cond, consequent))
                .map(|(condition, consequent)| RawConditionConsequent {
                    condition,
                    consequent,
                })
                .parse(input)
        };

    // We avoid a recursive parsing implementation of nested if...else if expressions to avoid
    // stack overflows when fuzzing.
    let ifelses =
        separated_list1(tag_ws("else"), spanned(parse_if_condition_consequent)).map(|xs| {
            let mut ys = NonEmpty::new(xs.first().unwrap().clone());
            ys.extend(xs.iter().skip(1).cloned());
            ys
        });

    let alternate = preceded(
        tag_ws("else"),
        delimited(tag_ws("{"), parse_expression(None), tag_ws("}")),
    );

    tuple((ifelses, alternate)).parse(input)
}

/// Parses unary and atomic expressions. E.g., `-u1:1`, `(u1:1 + u1:0)`
fn parse_unary_atomic_expression(input: ParseInput) -> ParseResult<Expression> {
    // this implementation follows the 'Top Down Operator Precedence' algorithm. See
    // <https://btmc.substack.com/p/how-to-parse-expressions-easy> or <https://tdop.github.io/>
    alt((
        spanned(delimited(
            tag("("),
            parse_expression(None).map(ParenthesizedExpression),
            tag_ws(")"),
        )),
        spanned(delimited(
            tag("{"),
            parse_expression(None).map(BlockExpression),
            tag_ws("}"),
        )),
        spanned(parse_let_expression), // TODO FIXME let is not an expression
        spanned(parse_ifelse_expression),
        spanned(tuple((parse_unary_operator, parse_unary_atomic_expression))),
        spanned(parse_literal),
        spanned(parse_identifier),
    ))
    .parse(input)
}

/// Parses a binary operator and the expression that follows it, given the expression preceding
/// the operator, returning all of this is a binary `Expression`.
///
/// E.g. left=`u1:1`, input=`&& u1:1`, returns the Expression for `u1:1 && u1:1`
///
/// parse_infix_expression handles any and all kinds of expressions that would
/// left-recursively call parse_expression
fn parse_infix_expression<'a>(
    left: Expression,
) -> impl FnMut(ParseInput<'a>) -> ParseResult<Expression> {
    // this implementation follows the 'Top Down Operator Precedence' algorithm. See
    // <https://btmc.substack.com/p/how-to-parse-expressions-easy> or <https://tdop.github.io/>
    //
    // If we had any right-associative operators, we would have to use the 'precedence
    // lowering' trick found here: https://btmc.substack.com/p/how-to-parse-expressions-easy
    // Like so: right = parse_expression(tokens, get_precedence(op.tag)-1)

    // Note: the spanned combinator can't be used here because `left` was passed to us (thus,
    // spanned can't capture the start).
    move |input: ParseInput<'a>| -> ParseResult<Expression> {
        flat_map(parse_binary_operator, |op| {
            tuple((
                success(left.clone()),
                success(op.clone()),
                parse_expression(Some(op.thing)),
            ))
        })
        .map(|(left, op, right)| {
            let span = Span {
                start: left.span.start.clone(),
                end: right.span.end.clone(),
            };
            let thing = RawExpression::from((left, op, right));
            Spanned { span, thing }
        })
        .parse(input)
    }
}

/// Parses an expression (e.g. binary, unary, arbitrarily nested expressions), given the
/// preceding binary operator (if one exists).
///
/// E.g. input=`u1:1 && u1:1`, previous=`Some(||)`, will return the `Expression` `u1:1 && u1:1`
/// because `&&` has higher precedence than `||`.
fn parse_expression<'a>(
    previous: Option<RawBinaryOperator>,
) -> impl FnMut(ParseInput<'a>) -> ParseResult<Expression> {
    // this implementation follows the 'Top Down Operator Precedence' algorithm. See
    // <https://btmc.substack.com/p/how-to-parse-expressions-easy> or <https://tdop.github.io/>

    move |input: ParseInput<'a>| {
        let higher_precedence_than_previous = |current: BinaryOperator| -> bool {
            match (previous, current.thing) {
                // when no previous, then we judge current is higher precedence
                (None, _) => true,
                (Some(previous), current) => match current.partial_cmp(&previous) {
                    Some(Greater) => true,
                    // no comparison, equal, and less are all judged not higher precedence
                    _ => false,
                },
            }
        };

        // Note: I could not figure out how to do the following using only combinators
        let (mut rest, mut left) = parse_unary_atomic_expression(input)?;
        let mut op: Option<BinaryOperator>;
        loop {
            (rest, op) = peek(opt(parse_binary_operator))(rest)?;
            match op {
                Some(op) => {
                    if higher_precedence_than_previous(op) {
                        (rest, left) = parse_infix_expression(left)(rest)?;
                    } else {
                        return Ok((rest, left));
                    }
                }

                // There is no binary operator following the unary/atomic expression; return now.
                None => return Ok((rest, left)),
            }
        }
    }
}

// TODO PR suggestion: move all tests to own file
#[cfg(test)]
mod tests {
    use super::*;
    use nom::combinator::all_consuming;
    use nom_locate::LocatedSpan;
    use num_traits::cast::FromPrimitive;

    // Panics if Expression is not the correct case
    fn expression_is_literal(x: Expression) -> RawLiteral {
        match x.thing {
            RawExpression::Literal(Spanned { span: _, thing }) => thing,
            e => panic!("wasn't Literal expression: {:?}", e),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_binding(x: Expression) -> RawIdentifier {
        match x.thing {
            RawExpression::Binding(Spanned { span: _, thing }) => thing,
            _ => panic!("wasn't Literal expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_unary(x: Expression) -> (RawUnaryOperator, Box<Expression>) {
        match x.thing {
            RawExpression::Unary(Spanned { span: _, thing }, expr) => (thing, expr),
            _ => panic!("wasn't Unary expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_binary(
        x: Expression,
    ) -> (Box<Expression>, RawBinaryOperator, Box<Expression>) {
        match x.thing {
            RawExpression::Binary(lhs, Spanned { span: _, thing: op }, rhs) => (lhs, op, rhs),
            _ => panic!("wasn't Binary expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_parenthesized(x: Expression) -> Box<Expression> {
        match x.thing {
            RawExpression::Parenthesized(b) => b,
            _ => panic!("wasn't Parenthesized expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_block(x: Expression) -> Box<Expression> {
        match x.thing {
            RawExpression::Block(b) => b,
            _ => panic!("wasn't Block expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_let(x: Expression) -> (NonEmpty<LetBinding>, Option<Box<Expression>>) {
        match x.thing {
            RawExpression::Let(xs, e) => (xs, e),
            _ => panic!("wasn't Let expression"),
        }
    }

    // Panics if Expression is not the correct case
    fn expression_is_ifelse(
        x: Expression,
    ) -> (NonEmpty<Box<ConditionConsequent>>, Box<Expression>) {
        match x.thing {
            RawExpression::IfElse(xs, e) => (xs, e),
            _ => panic!("wasn't ifelse expression"),
        }
    }

    #[test]
    fn test_consumes_ws() -> () {
        let parsed = match tag_ws("a").parse(ParseInput::new("a")) {
            Ok(x) => x.1,
            Err(_) => panic!(),
        };
        assert_eq!(parsed, unsafe {
            LocatedSpan::new_from_raw_offset(0, 1, "a", ())
        });

        let parsed = match tag_ws("a").parse(ParseInput::new(" a")) {
            Ok(x) => x.1,
            Err(_) => panic!(),
        };
        assert_eq!(parsed, unsafe {
            LocatedSpan::new_from_raw_offset(1, 1, "a", ())
        });

        let parsed = match tag_ws("a").parse(ParseInput::new("  a")) {
            Ok(x) => x.1,
            Err(_) => panic!(),
        };
        assert_eq!(parsed, unsafe {
            LocatedSpan::new_from_raw_offset(2, 1, "a", ())
        });
    }

    #[test]
    fn test_identifiers_raw_parser() -> () {
        match recognize(pair(
            alt((alpha1::<_, (_, nom::error::ErrorKind)>, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))
        .parse("_foo23Bar rest")
        {
            Ok(x) => assert_eq!(x, (" rest", "_foo23Bar")),
            Err(e) => {
                eprintln!("Error: {}", e);
                panic!()
            }
        };
    }

    #[test]
    fn test_parse_identifier() -> () {
        match parse_identifier(ParseInput::new(" _foo23Bar! ")) {
            Ok(x) => assert_eq!(
                x,
                (
                    unsafe { LocatedSpan::new_from_raw_offset(10, 1, "! ", (),) },
                    Spanned {
                        span: Span::from(((1, 1, 2), (10, 1, 11))),
                        thing: RawIdentifier("_foo23Bar".to_owned())
                    }
                ),
            ),
            Err(e) => {
                eprintln!("Error: {}", e);
                panic!()
            }
        };
    }

    #[test]
    fn test_parse_variable_declaration() -> () {
        let p = match parse_variable_declaration(ParseInput::new(" x : u2 ")) {
            Ok(x) => x.1,
            Err(e) => {
                eprintln!("Error: {}", e);
                panic!()
            }
        };
        assert_eq!(
            p,
            Spanned {
                span: Span::from(((1, 1, 2), (7, 1, 8))),
                thing: RawBindingDecl {
                    name: Spanned {
                        span: Span::from(((1, 1, 2), (2, 1, 3))),
                        thing: RawIdentifier("x".to_owned())
                    },
                    typ: Spanned {
                        span: Span::from(((5, 1, 6), (7, 1, 8))),
                        thing: RawIdentifier("u2".to_owned())
                    }
                }
            }
        );
    }

    #[test]
    fn test_parse_parameter_list0() -> () {
        let p = match parse_parameter_list0(ParseInput::new("x : u2,y : u4")) {
            Ok(x) => x.1,
            Err(e) => {
                eprintln!("Error: {}", e);
                panic!()
            }
        };
        assert_eq!(
            p,
            Spanned {
                span: Span::from(((0, 1, 1), (13, 1, 14))),
                thing: vec![
                    Spanned {
                        span: Span::from(((0, 1, 1), (6, 1, 7))),
                        thing: RawBindingDecl {
                            name: Spanned {
                                span: Span::from(((0, 1, 1), (1, 1, 2))),
                                thing: RawIdentifier("x".to_owned())
                            },
                            typ: Spanned {
                                span: Span::from(((4, 1, 5), (6, 1, 7))),
                                thing: RawIdentifier("u2".to_owned())
                            }
                        }
                    },
                    Spanned {
                        span: Span::from(((7, 1, 8), (13, 1, 14))),
                        thing: RawBindingDecl {
                            name: Spanned {
                                span: Span::from(((7, 1, 8), (8, 1, 9))),
                                thing: RawIdentifier("y".to_owned())
                            },
                            typ: Spanned {
                                span: Span::from(((11, 1, 12), (13, 1, 14))),
                                thing: RawIdentifier("u4".to_owned())
                            }
                        }
                    }
                ]
            }
        );
    }

    // TODO: Parse the rest of the fn.
    #[test]
    fn test_parse_fn_signature() -> () {
        let input = ParseInput::new("fn add_1(x: u32) -> u16");
        let expected = FunctionSignature {
            span: Span::from(((0, 1, 1), (23, 1, 24))),
            thing: RawFunctionSignature {
                name: Identifier {
                    span: Span::from(((3, 1, 4), (8, 1, 9))),
                    thing: RawIdentifier("add_1".to_owned()),
                },
                parameters: BindingDeclList {
                    span: Span::from(((9, 1, 10), (15, 1, 16))),
                    thing: vec![BindingDecl {
                        span: Span::from(((9, 1, 10), (15, 1, 16))),
                        thing: RawBindingDecl {
                            name: Identifier {
                                span: Span::from(((9, 1, 10), (10, 1, 11))),
                                thing: RawIdentifier("x".to_owned()),
                            },
                            typ: Identifier {
                                span: Span::from(((12, 1, 13), (15, 1, 16))),
                                thing: RawIdentifier("u32".to_owned()),
                            },
                        },
                    }],
                },
                result_type: Identifier {
                    span: Span::from(((20, 1, 21), (23, 1, 24))),
                    thing: RawIdentifier("u16".to_owned()),
                },
            },
        };
        let parsed = match parse_function_signature(input) {
            Ok(foo) => foo.1,
            Err(_) => panic!(),
        };
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_parse_unsigned_binary() -> () {
        // at least 1 digit is required
        parse_unsigned_binary(ParseInput::new("")).expect_err("");
        parse_unsigned_binary(ParseInput::new("0")).expect_err("");
        parse_unsigned_binary(ParseInput::new("0b")).expect_err("");

        // negative not accepted
        parse_unsigned_binary(ParseInput::new("-0b1")).expect_err("");

        // fractions not accepted
        parse_unsigned_binary(ParseInput::new("0b.1")).expect_err("");

        // Ensure that radix is 10
        parse_unsigned_binary(ParseInput::new("0b2")).expect_err("");

        // 0B invalid
        parse_unsigned_binary(ParseInput::new("0B1")).expect_err("");

        // rejects preceding whitespace
        let (_, num) = parse_unsigned_binary(ParseInput::new("0b1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        // Allow _ between digits...
        let (_, num) = parse_unsigned_binary(ParseInput::new("0b1_0_1_0")).unwrap();
        assert_eq!(num, BigUint::from_u128(8 + 2).unwrap());
        // but not leading _
        parse_unsigned_binary(ParseInput::new("_0b1")).expect_err("");
        parse_unsigned_binary(ParseInput::new("0b_1")).expect_err("");
        // and not trailing _
        // all_consuming tells us if parse_unsigned_binary consumed the _
        all_consuming(parse_unsigned_binary)(ParseInput::new("0b1_")).expect_err("");
        // and not more than 1 in a row
        all_consuming(parse_unsigned_binary)(ParseInput::new("0b1__0")).expect_err("");

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b11")).unwrap();
        assert_eq!(num, BigUint::from_u128(0b11).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b101")).unwrap();
        assert_eq!(num, BigUint::from_u128(0b101).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new(
            "0b1000000000000000000000000000000000000000000000000000000000000000000000000",
        ))
        .unwrap();
        assert_eq!(
            num,
            BigUint::from_u128(
                0b1000000000000000000000000000000000000000000000000000000000000000000000000
            )
            .unwrap()
        );

        // leading zeros are accepted
        let (_, num) = parse_unsigned_binary(ParseInput::new("0b0101")).unwrap();
        assert_eq!(num, BigUint::from_u128(0b0101).unwrap());
    }

    #[test]
    fn test_parse_unsigned_decimal() -> () {
        // at least 1 digit is required
        parse_unsigned_decimal(ParseInput::new("")).expect_err("");

        parse_unsigned_decimal(ParseInput::new("q0")).expect_err("");
        parse_unsigned_decimal(ParseInput::new(" q0")).expect_err("");

        // negative not accepted
        parse_unsigned_decimal(ParseInput::new("-1")).expect_err("");

        // fractions not accepted
        parse_unsigned_decimal(ParseInput::new(".1")).expect_err("");

        // Ensure that radix is 10
        parse_unsigned_decimal(ParseInput::new("a")).expect_err("");
        parse_unsigned_decimal(ParseInput::new("A")).expect_err("");

        // rejects preceding whitespace
        parse_unsigned_decimal(ParseInput::new("A")).expect_err("");

        // Allow _ between digits...
        let (_, num) = parse_unsigned_decimal(ParseInput::new("1_2_3_4_5_6_7_8_9")).unwrap();
        assert_eq!(num, BigUint::from_u128(123456789).unwrap());
        // but not leading _
        parse_unsigned_decimal(ParseInput::new("_12")).expect_err("");
        // and not trailing _
        // all_consuming tells us if parse_unsigned_decimal consumed the _
        all_consuming(parse_unsigned_decimal)(ParseInput::new("12_")).expect_err("");
        // and not more than 1 in a row
        all_consuming(parse_unsigned_decimal)(ParseInput::new("1__2")).expect_err("");

        // consumes all digits then stops at the first non digit
        let (rest, num) = parse_unsigned_decimal(ParseInput::new("0a ")).unwrap();
        assert_eq!(num, BigUint::from_u128(0).unwrap());
        assert_eq!(rest, unsafe {
            LocatedSpan::new_from_raw_offset(1, 1, "a ", ())
        });

        let (_, num) = parse_unsigned_decimal(ParseInput::new("1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        let (_, num) = parse_unsigned_decimal(ParseInput::new("95")).unwrap();
        assert_eq!(num, BigUint::from_u128(95).unwrap());

        let (_, num) = parse_unsigned_decimal(ParseInput::new("36893488147419103232")).unwrap();
        let two_tothe_65 = BigUint::from_u128(36893488147419103232).unwrap();
        assert_eq!(num, two_tothe_65);

        let (_, two_tothe_130) =
            parse_unsigned_decimal(ParseInput::new("1361129467683753853853498429727072845824"))
                .unwrap();
        assert_eq!(two_tothe_130, two_tothe_65.clone() * two_tothe_65.clone());
    }

    #[test]
    fn test_parse_unsigned_hexadecimal() -> () {
        // at least 1 digit is required
        parse_unsigned_hexadecimal(ParseInput::new("")).expect_err("");
        parse_unsigned_hexadecimal(ParseInput::new("0")).expect_err("");
        parse_unsigned_hexadecimal(ParseInput::new("0x")).expect_err("");

        // negative not accepted
        parse_unsigned_hexadecimal(ParseInput::new("-0x1")).expect_err("");

        // fractions not accepted
        parse_unsigned_hexadecimal(ParseInput::new("0x.1")).expect_err("");

        // Ensure that radix is 16
        parse_unsigned_hexadecimal(ParseInput::new("0xg")).expect_err("");
        parse_unsigned_hexadecimal(ParseInput::new("0xG")).expect_err("");

        // 0X invalid
        parse_unsigned_hexadecimal(ParseInput::new("0X1")).expect_err("");

        // rejects preceding whitespace
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0x1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        // Allow _ between digits...
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xa_b_c_d")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xabcd).unwrap());
        // but not leading _
        parse_unsigned_hexadecimal(ParseInput::new("_0x1")).expect_err("");
        parse_unsigned_hexadecimal(ParseInput::new("0x_1")).expect_err("");
        // and not trailing _
        // all_consuming tells us if parse_unsigned_hexadecimal consumed the _
        all_consuming(parse_unsigned_hexadecimal)(ParseInput::new("0x1_")).expect_err("");
        // and not more than 1 in a row
        all_consuming(parse_unsigned_hexadecimal)(ParseInput::new("0x1__0")).expect_err("");

        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0x0")).unwrap();
        assert_eq!(num, BigUint::from_u128(0).unwrap());

        // upper and lowercase hex digits accepted
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xff")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xff).unwrap());
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xfF")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xff).unwrap());
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xFf")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xff).unwrap());
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xFF")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xff).unwrap());

        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0xabcdef0123456789")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xabcdef0123456789).unwrap());

        // leading zeros are accepted
        let (_, num) = parse_unsigned_hexadecimal(ParseInput::new("0x0101")).unwrap();
        assert_eq!(num, BigUint::from_u128(0x0101).unwrap());
    }

    #[test]
    fn test_parse_unsigned_integer() -> () {
        // at least 1 digit is required
        parse_unsigned_integer(ParseInput::new("")).expect_err("");

        parse_unsigned_integer(ParseInput::new("0b0")).expect("");

        let (_, num) = parse_unsigned_integer(ParseInput::new("0b1101")).unwrap();
        assert_eq!(num, BigUint::from_u128(0b1101).unwrap());
        let (_, num) = parse_unsigned_integer(ParseInput::new("0b01101")).unwrap();
        assert_eq!(num, BigUint::from_u128(0b01101).unwrap());

        // Ensure that the 0 prefix doesn't cause a failure; the parse doesn't try hex or
        // binary then completely fail.
        let (_, num) = parse_unsigned_integer(ParseInput::new("0123456789")).unwrap();
        assert_eq!(num, BigUint::from_u128(123456789).unwrap());

        let (_, num) = parse_unsigned_integer(ParseInput::new("0xabcdef0123456789")).unwrap();
        assert_eq!(num, BigUint::from_u128(0xabcdef0123456789).unwrap());
    }

    #[test]
    fn test_parse_signed_integer() -> () {
        // at least 1 digit is required
        parse_signed_integer(ParseInput::new("")).expect_err("");

        // rejects preceding whitespace
        parse_signed_integer(ParseInput::new(" -b10")).expect_err("");

        // whitespace accepted after -
        let (_, num) = parse_signed_integer(ParseInput::new("- 0b10")).unwrap();
        assert_eq!(num, BigInt::from_i128(-2).unwrap());

        let (_, num) = parse_signed_integer(ParseInput::new("-0b10")).unwrap();
        assert_eq!(num, BigInt::from_i128(-2).unwrap());

        let (_, num) = parse_signed_integer(ParseInput::new("-3")).unwrap();
        assert_eq!(num, BigInt::from_i128(-3).unwrap());

        let (_, num) = parse_signed_integer(ParseInput::new("-0xF")).unwrap();
        assert_eq!(num, BigInt::from_i128(-15).unwrap());
    }

    #[test]
    fn parse_bit_type_shorthand() -> () {
        // must start with s or u
        parse_bit_type(ParseInput::new("")).expect_err("");
        parse_bit_type(ParseInput::new("1")).expect_err("");
        parse_bit_type(ParseInput::new("a")).expect_err("");

        // no spaces allowed inside the token u1 s1
        parse_bit_type(ParseInput::new(" s 1 ")).expect_err("");
        parse_bit_type(ParseInput::new(" u 1 ")).expect_err("");

        // whitespace accepted before
        parse_bit_type(ParseInput::new(" s1 ")).expect("");
        parse_bit_type(ParseInput::new(" u1 ")).expect("");

        // 0 bits is accepted
        parse_bit_type(ParseInput::new("s0")).expect("");
        parse_bit_type(ParseInput::new("u0")).expect("");

        let (_, r) = parse_bit_type(ParseInput::new("s1")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(1)
            }
        );

        let (_, r) = parse_bit_type(ParseInput::new("u1")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(1)
            }
        );

        // 64 is the largest
        parse_bit_type(ParseInput::new("u64")).expect("");
        parse_bit_type(ParseInput::new("s64")).expect("");
        parse_bit_type(ParseInput::new("u65")).expect_err("");
        parse_bit_type(ParseInput::new("s65")).expect_err("");
    }

    #[test]
    fn parse_bit_type_n_brackets() -> () {
        // no spaces allowed inside the token uN, sN
        parse_bit_type(ParseInput::new("u N[1]")).expect_err("");
        parse_bit_type(ParseInput::new("s N[1]")).expect_err("");

        // whitespace allowed between tokens
        parse_bit_type(ParseInput::new(" uN [ 1 ] ")).expect("");
        parse_bit_type(ParseInput::new(" sN [ 1 ] ")).expect("");

        // 0 bits is accepted
        parse_bit_type(ParseInput::new("sN[0]")).expect("");
        parse_bit_type(ParseInput::new("uN[0]")).expect("");

        let (_, r) = parse_bit_type(ParseInput::new("sN[1]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(1)
            }
        );

        let (_, r) = parse_bit_type(ParseInput::new("uN[1]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(1)
            }
        );

        // 2^(32)-1 is largest valid
        let (_, r) = parse_bit_type(ParseInput::new("sN[4294967295]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(4294967295)
            }
        );
        let (_, r) = parse_bit_type(ParseInput::new("uN[4294967295]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(4294967295)
            }
        );

        // 2^32 is too big
        parse_bit_type(ParseInput::new("uN[4294967296]")).expect_err("");
        parse_bit_type(ParseInput::new("sN[4294967296]")).expect_err("");
    }

    #[test]
    fn test_parse_bit_type_bits() -> () {
        // no spaces allowed inside the token `bits`
        parse_bit_type(ParseInput::new("b its[3]")).expect_err("");
        parse_bit_type(ParseInput::new("bi ts[3]")).expect_err("");
        parse_bit_type(ParseInput::new("bit s[3]")).expect_err("");

        // spaces allowed between tokens
        parse_bit_type(ParseInput::new("bits [3]")).expect("");
        parse_bit_type(ParseInput::new("bits[ 3]")).expect("");
        parse_bit_type(ParseInput::new("bits [ 3]")).expect("");
        parse_bit_type(ParseInput::new("bits[3 ]")).expect("");
        parse_bit_type(ParseInput::new("bits[ 3 ]")).expect("");
        parse_bit_type(ParseInput::new("bits [ 3 ] ")).expect("");

        // whitespace accepted before
        parse_bit_type(ParseInput::new(" bits[3] ")).expect("");

        // 0 bits is accepted
        parse_bit_type(ParseInput::new("bits[0]")).expect("");

        let (_, r) = parse_bit_type(ParseInput::new("bits[1]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(1)
            }
        );

        // 2^(32)-1 is largest valid
        let (_, r) = parse_bit_type(ParseInput::new("bits[4294967295]")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(4294967295)
            }
        );

        // 2^32 is too big
        parse_bit_type(ParseInput::new("bits[4294967296]")).expect_err("");
    }

    #[test]
    fn test_parse_literal() -> () {
        // incomplete is rejected
        parse_literal(ParseInput::new(":12")).expect_err("");
        parse_literal(ParseInput::new("u8 12")).expect_err("");
        parse_literal(ParseInput::new("u8:")).expect_err("");

        // spaces allowed between tokens
        parse_literal(ParseInput::new(" u8:12 ")).expect("");
        parse_literal(ParseInput::new(" u8 :12 ")).expect("");
        parse_literal(ParseInput::new(" u8 : 12 ")).expect("");
        parse_literal(ParseInput::new(" s8 :- 128 ")).expect("");
        parse_literal(ParseInput::new(" s8 : -128 ")).expect("");
        parse_literal(ParseInput::new(" u8 : 128 ")).expect("");
        parse_literal(ParseInput::new(" s8 : 128 ")).expect("");
        parse_literal(ParseInput::new(" s8 : - 128 ")).expect("");
        parse_literal(ParseInput::new(" u8 : 0xf ")).expect("");
        parse_literal(ParseInput::new(" s8 : 0xf ")).expect("");
        parse_literal(ParseInput::new(" s8 : - 0xf ")).expect("");
        parse_literal(ParseInput::new(" u8 : 0b1 ")).expect("");
        parse_literal(ParseInput::new(" s8 : 0b1 ")).expect("");
        parse_literal(ParseInput::new(" s8 : - 0b1 ")).expect("");

        // 0 bits is accepted
        parse_literal(ParseInput::new(" u0 : 0 ")).expect("");

        // unsigned shorthand
        let (_, r) = parse_literal(ParseInput::new("u8:3")).unwrap();
        assert_eq!(
            r.thing.value.thing,
            ast::RawInteger::Unsigned(BigUint::from_u128(3).unwrap())
        );
        assert_eq!(
            r.thing.bit_type.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(8)
            }
        );

        // signed shorthand
        let (_, r) = parse_literal(ParseInput::new("s51:-5")).unwrap();
        assert_eq!(
            r.thing.value.thing,
            ast::RawInteger::Signed(BigInt::from_i128(-5).unwrap())
        );
        assert_eq!(
            r.thing.bit_type.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(51)
            }
        );

        // uN[]
        let (_, r) = parse_literal(ParseInput::new("uN[13]:2")).unwrap();
        assert_eq!(
            r.thing.value.thing,
            ast::RawInteger::Unsigned(BigUint::from_u128(2).unwrap())
        );
        assert_eq!(
            r.thing.bit_type.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(13)
            }
        );

        // sN[]
        let (_, r) = parse_literal(ParseInput::new("sN[7]:-9")).unwrap();
        assert_eq!(
            r.thing.value.thing,
            ast::RawInteger::Signed(BigInt::from_i128(-9).unwrap())
        );
        assert_eq!(
            r.thing.bit_type.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(7)
            }
        );

        // bits[]
        let (_, r) = parse_literal(ParseInput::new("bits[17]:11")).unwrap();
        assert_eq!(
            r.thing.value.thing,
            ast::RawInteger::Unsigned(BigUint::from_u128(11).unwrap())
        );
        assert_eq!(
            r.thing.bit_type.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(17)
            }
        );

        // 2^32 is too big
        parse_literal(ParseInput::new("bits[4294967296]:1")).expect_err("");
    }

    #[test]
    fn test_parse_unary_operator() -> () {
        // spaces allowed between tokens
        parse_unary_operator(ParseInput::new(" - ")).expect("");
        parse_unary_operator(ParseInput::new(" ! ")).expect("");

        let (_, r) = parse_unary_operator(ParseInput::new("-")).unwrap();
        assert_eq!(r.thing, RawUnaryOperator::Negate);

        let (_, r) = parse_unary_operator(ParseInput::new("!")).unwrap();
        assert_eq!(r.thing, RawUnaryOperator::Invert);
    }

    #[test]
    fn test_parse_unary_atomic_expression() -> () {
        // a lone operator is not an expression
        all_consuming(parse_unary_atomic_expression)(ParseInput::new("!")).expect_err("");
        all_consuming(parse_unary_atomic_expression)(ParseInput::new("-")).expect_err("");

        // literals match
        all_consuming(parse_unary_atomic_expression)(ParseInput::new("u1:1")).expect("");

        // unary expressions match
        all_consuming(parse_unary_atomic_expression)(ParseInput::new("!u1:1")).expect("");
        all_consuming(parse_unary_atomic_expression)(ParseInput::new("-u1:1")).expect("");

        // accepts whitespace
        all_consuming(parse_unary_atomic_expression)(ParseInput::new(" - u1 : 1")).expect("");

        let (_, r) =
            all_consuming(parse_unary_atomic_expression)(ParseInput::new("-u1:1")).unwrap();
        let (op, inner_expr) = expression_is_unary(r);
        assert_eq!(op, RawUnaryOperator::Negate);
        let _ = expression_is_literal(*inner_expr);

        let (_, r) =
            all_consuming(parse_unary_atomic_expression)(ParseInput::new("!u1:1")).unwrap();
        let (op, inner_expr) = expression_is_unary(r);
        assert_eq!(op, RawUnaryOperator::Invert);
        let _ = expression_is_literal(*inner_expr);

        // negate is the outer expression
        let (_, r) =
            all_consuming(parse_unary_atomic_expression)(ParseInput::new("-!u1:1")).unwrap();
        let (op, inner_expr) = expression_is_unary(r);
        assert_eq!(op, RawUnaryOperator::Negate);
        let (op, inner_expr) = expression_is_unary(*inner_expr);
        assert_eq!(op, RawUnaryOperator::Invert);
        let _ = expression_is_literal(*inner_expr);
    }

    #[test]
    fn test_parse_binary_operator() -> () {
        // spaces allowed between tokens
        parse_binary_operator(ParseInput::new(" | ")).expect("");
        parse_binary_operator(ParseInput::new(" & ")).expect("");

        let (_, r) = parse_binary_operator(ParseInput::new("||")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::BooleanOr);
        let (_, r) = parse_binary_operator(ParseInput::new("&&")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::BooleanAnd);

        let (_, r) = parse_binary_operator(ParseInput::new("|")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::BitwiseOr);
        let (_, r) = parse_binary_operator(ParseInput::new("&")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::BitwiseAnd);
        let (_, r) = parse_binary_operator(ParseInput::new("^")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::BitwiseXor);

        let (_, r) = parse_binary_operator(ParseInput::new("+")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Add);
        let (_, r) = parse_binary_operator(ParseInput::new("-")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Subtract);
        let (_, r) = parse_binary_operator(ParseInput::new("*")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Multiply);

        let (_, r) = parse_binary_operator(ParseInput::new(">>")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::ShiftRight);
        let (_, r) = parse_binary_operator(ParseInput::new("<<")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::ShiftLeft);

        let (_, r) = parse_binary_operator(ParseInput::new("==")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Equal);
        let (_, r) = parse_binary_operator(ParseInput::new("!=")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::NotEqual);
        let (_, r) = parse_binary_operator(ParseInput::new(">=")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::GreaterOrEqual);
        let (_, r) = parse_binary_operator(ParseInput::new(">")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Greater);
        let (_, r) = parse_binary_operator(ParseInput::new("<=")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::LessOrEqual);
        let (_, r) = parse_binary_operator(ParseInput::new("<")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Less);

        let (_, r) = parse_binary_operator(ParseInput::new("++")).unwrap();
        assert_eq!(r.thing, RawBinaryOperator::Concatenate);
    }

    #[test]
    fn test_parse_binary_expression_basic() -> () {
        // spaces allowed between tokens
        parse_expression(None)(ParseInput::new(" u2 : 1 + u2 : 2 ")).expect("");
        parse_expression(None)(ParseInput::new(" u1 : 1 || u1 : 0 ")).expect("");

        // no space inside the || token
        parse_expression(None)(ParseInput::new(" u1 : 1 | | u1 : 0 ")).expect_err("");

        // unary expressions match
        let (_, r) = all_consuming(parse_expression(None))(ParseInput::new("-u1:1")).unwrap();
        let (op, inner_expr) = expression_is_unary(r);
        assert_eq!(op, RawUnaryOperator::Negate);
        let _ = expression_is_literal(*inner_expr);

        // match a few different binary operators
        let (lhs, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new("u1:0 * u2:1"))
                .unwrap()
                .1,
        );
        assert_eq!(op, RawBinaryOperator::Multiply);
        let _ = expression_is_literal(*lhs);
        let _ = expression_is_literal(*rhs);

        let (lhs, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new("u1:0 + u2:1"))
                .unwrap()
                .1,
        );
        assert_eq!(op, RawBinaryOperator::Add);
        let _ = expression_is_literal(*lhs);
        let _ = expression_is_literal(*rhs);

        let (lhs, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new("u1:0 | u2:1"))
                .unwrap()
                .1,
        );
        assert_eq!(op, RawBinaryOperator::BitwiseOr);
        let _ = expression_is_literal(*lhs);
        let _ = expression_is_literal(*rhs);

        let (lhs, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new("u1:0 || u2:1"))
                .unwrap()
                .1,
        );
        assert_eq!(op, RawBinaryOperator::BooleanOr);
        let _ = expression_is_literal(*lhs);
        let _ = expression_is_literal(*rhs);
    }

    /// Is the operation that evalutes first in the LHS or RHS of the second operation?
    enum FirstsLocation {
        // matches a pattern like
        //     s
        //    / \
        //   f   z
        //  / \
        // x   y
        LeftHandSide,
        // matches a pattern like
        //     s
        //    / \
        //   z  f
        //     / \
        //    x   y
        RightHandSide,
    }

    // Tests parsing of expressions containing 2 (or 3) binary operators, asserts that
    // precedence is correctly reflected in the AST
    #[test]
    fn test_parse_binary_expression_precedence() -> () {
        // Tests an expression containing two binary operators. Asserts that `first` is the
        // operation that will be evaluated first, followed by `second`
        fn first_then(
            s: &str,
            first: RawBinaryOperator,
            second: RawBinaryOperator,
            loc: FirstsLocation,
        ) -> () {
            let (lhs, op, rhs) = expression_is_binary(
                all_consuming(parse_expression(None))(ParseInput::new(s))
                    .unwrap()
                    .1,
            );

            // The operation that occurs second will be outermost
            assert_eq!(op, second);

            match loc {
                FirstsLocation::LeftHandSide => {
                    let _ = expression_is_literal(*rhs);
                    let (lhs, op, rhs) = expression_is_binary(*lhs);
                    assert_eq!(op, first);
                    let _ = expression_is_literal(*lhs);
                    let _ = expression_is_literal(*rhs);
                }
                FirstsLocation::RightHandSide => {
                    let _ = expression_is_literal(*lhs);
                    let (lhs, op, rhs) = expression_is_binary(*rhs);
                    assert_eq!(op, first);
                    let _ = expression_is_literal(*lhs);
                    let _ = expression_is_literal(*rhs);
                }
            }
        }

        // multiply is highest
        first_then(
            "u1:1 * u2:2 + u3:3",
            RawBinaryOperator::Multiply,
            RawBinaryOperator::Add,
            FirstsLocation::LeftHandSide,
        );
        first_then(
            "u1:1 + u1:1 * u1:1",
            RawBinaryOperator::Multiply,
            RawBinaryOperator::Add,
            FirstsLocation::RightHandSide,
        );
        first_then(
            "u1:1 * u1:1 || u1:1",
            RawBinaryOperator::Multiply,
            RawBinaryOperator::BooleanOr,
            FirstsLocation::LeftHandSide,
        );

        // add and subtract are same precedence, left associative
        first_then(
            "u1:1 + u1:1 - u1:1",
            RawBinaryOperator::Add,
            RawBinaryOperator::Subtract,
            FirstsLocation::LeftHandSide,
        );
        first_then(
            "u1:1 - u2:2 + u3:3",
            RawBinaryOperator::Subtract,
            RawBinaryOperator::Add,
            FirstsLocation::LeftHandSide,
        );

        // boolean AND > boolean OR
        first_then(
            "u1:1 || u1:1 && u1:1",
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
            FirstsLocation::RightHandSide,
        );
        first_then(
            "u1:1 && u1:1 || u1:1",
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
            FirstsLocation::LeftHandSide,
        );

        // mul and add in parallel, combine with bitwise AND
        let s = "u1:1 * u1:1 & u1:1 + u1:1";
        let (lhs, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new(s))
                .unwrap()
                .1,
        );

        // The operation that occurs last will be outermost
        assert_eq!(op, RawBinaryOperator::BitwiseAnd);

        {
            let (lhs, op, rhs) = expression_is_binary(*lhs);
            assert_eq!(op, RawBinaryOperator::Multiply);
            let _ = expression_is_literal(*lhs);
            let _ = expression_is_literal(*rhs);
        }
        {
            let (lhs, op, rhs) = expression_is_binary(*rhs);
            assert_eq!(op, RawBinaryOperator::Add);
            let _ = expression_is_literal(*lhs);
            let _ = expression_is_literal(*rhs);
        }

        // range operator is lower precedence than boolean or
        first_then(
            "u1:0 || u1:0 .. u1:1",
            RawBinaryOperator::BooleanOr,
            RawBinaryOperator::Range,
            FirstsLocation::LeftHandSide,
        );
    }

    // Tests parsing of expressions containing (), asserts that
    // precedence is correctly reflected in the AST
    #[test]
    fn test_parse_parenthesized_expression_precedence() -> () {
        // Tests an expression containing two binary operators. Asserts that `first` is the
        // operation that will be evaluated first, followed by `second`
        fn first_then(
            s: &str,
            first: RawBinaryOperator,
            second: RawBinaryOperator,
            loc: FirstsLocation,
        ) -> () {
            let (lhs, op, rhs) = expression_is_binary(
                all_consuming(parse_expression(None))(ParseInput::new(s))
                    .unwrap()
                    .1,
            );

            // The operation that occurs second will be outermost
            assert_eq!(op, second);
            match loc {
                FirstsLocation::LeftHandSide => {
                    let _ = expression_is_literal(*rhs);
                    let (lhs, op, rhs) = expression_is_binary(*expression_is_parenthesized(*lhs));
                    assert_eq!(op, first);
                    let _ = expression_is_literal(*lhs);
                    let _ = expression_is_literal(*rhs);
                }
                FirstsLocation::RightHandSide => {
                    let _ = expression_is_literal(*lhs);
                    let (lhs, op, rhs) = expression_is_binary(*expression_is_parenthesized(*rhs));
                    assert_eq!(op, first);
                    let _ = expression_is_literal(*lhs);
                    let _ = expression_is_literal(*rhs);
                }
            }
        }

        // mismatched/interleaved parens is wrong
        parse_expression(None)(ParseInput::new("(u1:1(+)u1:1)")).expect_err("");
        parse_expression(None)(ParseInput::new("(u1:1(+u1:1))")).expect_err("");
        // fix the above
        parse_expression(None)(ParseInput::new("((u1:1+u1:1))")).expect("");

        // whitespace accepted
        all_consuming(parse_expression(None))(ParseInput::new(" ( u1:1+u1:1 )")).expect("");

        // ordinarily, add is lower prec.
        first_then(
            "u1:1 * (u2:2 + u3:3)",
            RawBinaryOperator::Add,
            RawBinaryOperator::Multiply,
            FirstsLocation::RightHandSide,
        );
        first_then(
            "(u1:1 + u1:1) * u1:1",
            RawBinaryOperator::Add,
            RawBinaryOperator::Multiply,
            FirstsLocation::LeftHandSide,
        );

        // ordinarily, boolean OR is lowest prec.
        first_then(
            "u1:1 * (u1:1 || u1:1)",
            RawBinaryOperator::BooleanOr,
            RawBinaryOperator::Multiply,
            FirstsLocation::RightHandSide,
        );
        first_then(
            "(u1:1 || u1:1) * u1:1",
            RawBinaryOperator::BooleanOr,
            RawBinaryOperator::Multiply,
            FirstsLocation::LeftHandSide,
        );

        // add and subtract are same precedence, ordinarily left associative
        first_then(
            "u1:1 + (u1:1 - u1:1)",
            RawBinaryOperator::Subtract,
            RawBinaryOperator::Add,
            FirstsLocation::RightHandSide,
        );
        first_then(
            "u1:1 - (u2:2 + u3:3)",
            RawBinaryOperator::Add,
            RawBinaryOperator::Subtract,
            FirstsLocation::RightHandSide,
        );
        // now force left association
        first_then(
            "(u1:1 + u1:1) - u1:1",
            RawBinaryOperator::Add,
            RawBinaryOperator::Subtract,
            FirstsLocation::LeftHandSide,
        );
        first_then(
            "(u1:1 - u2:2) + u3:3",
            RawBinaryOperator::Subtract,
            RawBinaryOperator::Add,
            FirstsLocation::LeftHandSide,
        );

        // two parens changes nothing
        {
            let first = RawBinaryOperator::BooleanOr;
            let (_lhs, _, rhs) = expression_is_binary(
                all_consuming(parse_expression(None))(ParseInput::new("u1:1 * ((u1:1 || u1:1))"))
                    .unwrap()
                    .1,
            );

            let expr = expression_is_parenthesized(*rhs);
            let (lhs, op, rhs) = expression_is_binary(*expression_is_parenthesized(*expr));
            let _ = expression_is_literal(*lhs);
            let _ = expression_is_literal(*rhs);
            assert_eq!(op, first);
        };
    }

    // Test that spans are correct
    #[test]
    fn test_parse_binary_expression_span() -> () {
        let (_, r) =
            all_consuming(parse_expression(None))(ParseInput::new(" u1:0 * u2:1")).unwrap();
        assert_eq!(r.span, Span::from(((1, 1, 2), (12, 1, 13))));

        let s = " u1:1 * u1:1 & u1:1 + u1:1";
        let (lhs, _, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new(s))
                .unwrap()
                .1,
        );
        assert_eq!((*lhs).span, Span::from(((1, 1, 2), (12, 1, 13))));
        assert_eq!((*rhs).span, Span::from(((15, 1, 16), (26, 1, 27))));
    }

    #[test]
    fn test_parse_block_expression() -> () {
        // mismatched/interleaved curly {} is wrong
        all_consuming(parse_expression(None))(ParseInput::new("{u1:1")).expect_err("");
        all_consuming(parse_expression(None))(ParseInput::new("u1:1}")).expect_err("");
        all_consuming(parse_expression(None))(ParseInput::new("{{u1:1}")).expect_err("");
        all_consuming(parse_expression(None))(ParseInput::new("{u1:1}}")).expect_err("");
        all_consuming(parse_expression(None))(ParseInput::new("{{u1}:1}}")).expect_err("");
        all_consuming(parse_expression(None))(ParseInput::new("u1{:}1")).expect_err("");

        // fix the above
        all_consuming(parse_expression(None))(ParseInput::new("{u1:1}")).expect("");

        // whitespace accepted
        all_consuming(parse_expression(None))(ParseInput::new(" { u1 : 1 }")).expect("");

        let inside = expression_is_block(
            parse_expression(None)(ParseInput::new("{ let a: u32 = u32:1 * u32:2; a & a }"))
                .unwrap()
                .1,
        );
        expression_is_let(*inside);
    }

    #[test]
    fn test_parse_let_expression() -> () {
        // whitespace accepted
        all_consuming(parse_expression(None))(ParseInput::new("let  foo : u32 = bar;")).expect("");

        // test the first variable decl, and the using expression
        let s = r"let a: u32 = u32:1 * u32:2;
        a & a";
        let (bindings, using_expr) = expression_is_let(
            all_consuming(parse_expression(None))(ParseInput::new(s))
                .unwrap()
                .1,
        );
        assert_eq!(
            bindings.first().thing.variable_declaration.thing.name.thing,
            RawIdentifier("a".to_owned())
        );
        assert_eq!(
            bindings.first().thing.variable_declaration.thing.typ.thing,
            RawIdentifier("u32".to_owned())
        );
        let (_, op, _) = expression_is_binary(*bindings.first().thing.value.clone());
        assert_eq!(op, RawBinaryOperator::Multiply);
        let (_, op, _) = expression_is_binary(*using_expr.unwrap());
        assert_eq!(op, RawBinaryOperator::BitwiseAnd);

        // test two bindings, and no using expression.
        let s = r"let b: u16 = u16:1 + u16:2;
        let c: u8 = u16:3;";
        let (bindings, using_expr) = expression_is_let(
            all_consuming(parse_expression(None))(ParseInput::new(s))
                .unwrap()
                .1,
        );
        assert_eq!(
            bindings[0].thing.variable_declaration.thing.name.thing,
            RawIdentifier("b".to_owned())
        );
        assert_eq!(
            bindings[0].thing.variable_declaration.thing.typ.thing,
            RawIdentifier("u16".to_owned())
        );
        assert_eq!(
            bindings[1].thing.variable_declaration.thing.name.thing,
            RawIdentifier("c".to_owned())
        );
        assert_eq!(
            bindings[1].thing.variable_declaration.thing.typ.thing,
            RawIdentifier("u8".to_owned())
        );
        let (_, op, _) = expression_is_binary(*bindings.first().thing.value.clone());
        assert_eq!(op, RawBinaryOperator::Add);
        let _ = expression_is_literal(*bindings[1].thing.value.clone());
        assert_eq!(using_expr, None);
    }

    #[test]
    fn test_parse_ifelse_expression() -> () {
        // basic
        all_consuming(parse_ifelse_expression)(ParseInput::new(
            "if condition {whentrue} else {whenfalse}",
        ))
        .expect("");

        // whitespace accepted
        all_consuming(parse_ifelse_expression)(ParseInput::new(
            " if condition { whentrue } else { whenfalse }",
        ))
        .expect("");

        // only necessary whitespace
        all_consuming(parse_ifelse_expression)(ParseInput::new(
            "if condition{whentrue}else{whenfalse}",
        ))
        .expect("");

        // if requires trailing whitespace
        all_consuming(parse_ifelse_expression)(ParseInput::new(
            "ifcondition{whentrue}else{whenfalse}",
        ))
        .expect_err("");

        let (condition_consequent, alternate) = expression_is_ifelse(
            all_consuming(parse_expression(None))(ParseInput::new(
                "if condition {whentrue} else {whenfalse}",
            ))
            .unwrap()
            .1,
        );
        assert_eq!(condition_consequent.len(), 1);
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.condition.clone()),
            RawIdentifier("condition".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.consequent.clone()),
            RawIdentifier("whentrue".to_owned())
        );
        assert_eq!(
            expression_is_binding(*alternate),
            RawIdentifier("whenfalse".to_owned())
        );

        // an if expression inside another expression
        let (_, op, rhs) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new(
                "u1:1 + if condition {whentrue} else {whenfalse}",
            ))
            .unwrap()
            .1,
        );
        assert_eq!(op, RawBinaryOperator::Add);
        let (condition_consequent, alternate) = expression_is_ifelse(*rhs);
        assert_eq!(condition_consequent.len(), 1);
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.condition.clone()),
            RawIdentifier("condition".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.consequent.clone()),
            RawIdentifier("whentrue".to_owned())
        );
        assert_eq!(
            expression_is_binding(*alternate),
            RawIdentifier("whenfalse".to_owned())
        );

        // an if expression on the lhs of another expression
        let (lhs, op, _) = expression_is_binary(
            all_consuming(parse_expression(None))(ParseInput::new(
                "if condition {whentrue} else {whenfalse} + u1:1",
            ))
            .unwrap()
            .1,
        );
        assert_eq!(op, RawBinaryOperator::Add);
        let (condition_consequent, alternate) = expression_is_ifelse(*lhs);
        assert_eq!(condition_consequent.len(), 1);
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.condition.clone()),
            RawIdentifier("condition".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.consequent.clone()),
            RawIdentifier("whentrue".to_owned())
        );
        assert_eq!(
            expression_is_binding(*alternate),
            RawIdentifier("whenfalse".to_owned())
        );
    }

    #[test]
    fn test_parse_if_elseif_else_expression() -> () {
        let (condition_consequent, alternate) = expression_is_ifelse(
            all_consuming(parse_expression(None))(ParseInput::new(
                "if condition {whentrue} else if condition2 {whentrue2} else {whenfalse}",
            ))
            .unwrap()
            .1,
        );
        assert_eq!(condition_consequent.len(), 2);
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.condition.clone()),
            RawIdentifier("condition".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[0]).thing.consequent.clone()),
            RawIdentifier("whentrue".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[1]).thing.condition.clone()),
            RawIdentifier("condition2".to_owned())
        );
        assert_eq!(
            expression_is_binding((*condition_consequent[1]).thing.consequent.clone()),
            RawIdentifier("whentrue2".to_owned())
        );
        assert_eq!(
            expression_is_binding(*alternate),
            RawIdentifier("whenfalse".to_owned())
        );
    }
}
