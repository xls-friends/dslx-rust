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

pub mod ast;

use ast::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::hex_digit1,
    character::complete::{alpha1, alphanumeric1, char, digit1},
    combinator::verify,
    combinator::{map_opt, map_res, not, opt, peek, recognize, value},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult, Parser,
};
use num_bigint::{BigInt, BigUint};

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

/// Parses a single param, e.g., `x: u32`.
fn parse_param(input: ParseInput) -> ParseResult<Parameter> {
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
fn parse_literal(input: ParseInput) -> ParseResult<Literal> {
    fn parse(input: ParseInput) -> ParseResult<RawLiteral> {
        let (input, bit_type) = terminated(parse_bit_type, tag_ws(":"))(input)?;
        let (rest, value): (ParseInput, Integer) = match bit_type.thing.signedness {
            Signedness::Signed => spanned(parse_signed_integer).parse(input),
            Signedness::Unsigned => spanned(parse_unsigned_integer).parse(input),
        }?;
        Ok((rest, RawLiteral { value, bit_type }))
    }

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

/// Parses a unary expression. E.g., `-u1:1`, `!u1:1`
fn parse_unary_expression(input: ParseInput) -> ParseResult<(UnaryOperator, Expression)> {
    tuple((parse_unary_operator, parse_expression)).parse(input)
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
    ));
    spanned(op).parse(input)
}

/// Parses an expression. E.g.,
///
/// `u1:1`, `!u1:1`, `!!u1:1`
fn parse_expression(input: ParseInput) -> ParseResult<Expression> {
    alt((spanned(parse_literal), spanned(parse_unary_expression))).parse(input)
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use nom::combinator::all_consuming;
    use nom_locate::LocatedSpan;

    use num_traits::cast::FromPrimitive;

    use crate::ast::{
        Parameter, RawBitType, RawExpression, RawFunctionSignature, RawIdentifier, RawParameter,
        Usize,
    };

    use super::*;

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
                        thing: RawIdentifier { name: "_foo23Bar" }
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
    fn test_parse_param() -> () {
        let p = match parse_param(ParseInput::new(" x : u2 ")) {
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
                thing: RawParameter {
                    name: Spanned {
                        span: Span::from(((1, 1, 2), (2, 1, 3))),
                        thing: RawIdentifier { name: "x" }
                    },
                    param_type: Spanned {
                        span: Span::from(((5, 1, 6), (7, 1, 8))),
                        thing: RawIdentifier { name: "u2" }
                    }
                }
            }
        );
    }

    #[test]
    fn test_parse_param_list2() -> () {
        let p = match parse_param_list0(ParseInput::new("x : u2,y : u4")) {
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
                        thing: RawParameter {
                            name: Spanned {
                                span: Span::from(((0, 1, 1), (1, 1, 2))),
                                thing: RawIdentifier { name: "x" }
                            },
                            param_type: Spanned {
                                span: Span::from(((4, 1, 5), (6, 1, 7))),
                                thing: RawIdentifier { name: "u2" }
                            }
                        }
                    },
                    Spanned {
                        span: Span::from(((7, 1, 8), (13, 1, 14))),
                        thing: RawParameter {
                            name: Spanned {
                                span: Span::from(((7, 1, 8), (8, 1, 9))),
                                thing: RawIdentifier { name: "y" }
                            },
                            param_type: Spanned {
                                span: Span::from(((11, 1, 12), (13, 1, 14))),
                                thing: RawIdentifier { name: "u4" }
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
    fn test_parse_unary_expression() -> () {
        // a lone operator is not an expression
        all_consuming(parse_unary_expression)(ParseInput::new("!")).expect_err("");
        all_consuming(parse_unary_expression)(ParseInput::new("-")).expect_err("");

        // literal is not unary expression
        all_consuming(parse_unary_expression)(ParseInput::new("u1:1")).expect_err("");

        all_consuming(parse_unary_expression)(ParseInput::new("!u1:1")).expect("");
        all_consuming(parse_unary_expression)(ParseInput::new("-u1:1")).expect("");

        // accepts whitespace
        all_consuming(parse_unary_expression)(ParseInput::new(" - u1 : 1")).expect("");

        let (_, r) = all_consuming(parse_unary_expression)(ParseInput::new("-u1:1")).unwrap();
        assert_eq!(r.0.thing, RawUnaryOperator::Negate);
        assert_matches!(r.1.thing, RawExpression::Literal(_));

        let (_, r) = all_consuming(parse_unary_expression)(ParseInput::new("!u1:1")).unwrap();
        assert_eq!(r.0.thing, RawUnaryOperator::Invert);
        assert_matches!(r.1.thing, RawExpression::Literal(_));

        // negate is the outer expression
        let (_, r) = all_consuming(parse_unary_expression)(ParseInput::new("-!u1:1")).unwrap();
        assert_eq!(r.0.thing, RawUnaryOperator::Negate);
        assert_matches!(r.1.thing, RawExpression::Unary(_, _));
        if let RawExpression::Unary(Spanned { span: _, thing }, _) = r.1.thing {
            assert_matches!(thing, RawUnaryOperator::Invert);
        };
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
}
