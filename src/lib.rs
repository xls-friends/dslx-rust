//! **WIP** parser for the DSLX language.
//!
//! Full language defined here: <https://google.github.io/xls/dslx_reference/>
//! Currently a spooky scary skeleton, but actively being built out.
//!
//! At present, the _only_ entry point is `parse_function_signature`, taking in an ast::ParseInput.
pub mod ast;

use ast::{
    BitType, FunctionSignature, Identifier, Parameter, ParameterList, ParseInput, Span, Spanned,
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{alpha1, alphanumeric1, char, digit1},
    combinator::{map_opt, map_res, opt, recognize, verify},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, tuple},
    IResult, Parser,
};
use num_bigint::{BigInt, BigUint};

use crate::ast::Signedness;

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
/// `0b10`
///
/// `0b0011`
fn parse_unsigned_binary(input: ParseInput) -> ParseResult<BigUint> {
    let prefix = tag_ws("0b");
    let digits = take_while(|c: char| c == '0' || c == '1');
    map_opt(preceded(prefix, digits), |s| {
        BigUint::parse_bytes(s.fragment().as_bytes(), 2)
    })
    .parse(input)
}

/// Parses an unsigned decimal integer of arbitrary length, e.g.:
///
/// `0`
///
/// `1`
///
/// `42`
///
/// `1361129467683753853853498429727072845824`
fn parse_unsigned_decimal(input: ParseInput) -> ParseResult<BigUint> {
    let digits = preceding_whitespace(digit1);
    map_opt(digits, |s| {
        BigUint::parse_bytes(s.fragment().as_bytes(), 10)
    })
    .parse(input)
}

/// Parses a signed decimal integer of arbitrary length, e.g.:
///
/// `0`
///
/// `-1`
///
/// `42`
///
/// `1361129467683753853853498429727072845824`
fn parse_signed_decimal(input: ParseInput) -> ParseResult<BigInt> {
    let negative = opt(preceding_whitespace(char('-')));
    nom::combinator::map(
        tuple((negative, parse_unsigned_decimal)),
        |(neg, bu)| match neg {
            Some(_) => BigInt::from_biguint(num_bigint::Sign::Minus, bu),
            None => BigInt::from_biguint(num_bigint::Sign::Plus, bu),
        },
    )
    .parse(input)
}

/// Parses a shorthand bit type. E.g.:
///
/// `u1` `u16`
///
/// `s8` `s63`
///
/// See <https://google.github.io/xls/dslx_reference/#bit-type>
fn parse_bit_type_shorthand(input: ParseInput) -> ParseResult<BitType> {
    let sign = preceding_whitespace(alt((
        char('s').map(|_| Signedness::Signed),
        char('u').map(|_| Signedness::Unsigned),
    )));
    let width = verify(
        map_res(digit1, |s: ParseInput| s.fragment().parse::<u32>()),
        // "These are defined up to u64."
        |&width| width <= 64,
    );
    spanned(tuple((sign, width))).parse(input)
}

#[cfg(test)]
mod tests {
    use nom_locate::LocatedSpan;

    use num_traits::cast::FromPrimitive;

    use crate::ast::{
        Parameter, RawBitType, RawFunctionSignature, RawIdentifier, RawParameter, Usize,
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

        // accepts whitespace
        let (_, num) = parse_unsigned_binary(ParseInput::new(" 0b1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b1")).unwrap();
        assert_eq!(num, BigUint::from_u128(1).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b11")).unwrap();
        assert_eq!(num, BigUint::from_u128(3).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new("0b101")).unwrap();
        assert_eq!(num, BigUint::from_u128(4 + 1).unwrap());

        let (_, num) = parse_unsigned_binary(ParseInput::new(
            "0b1000000000000000000000000000000000000000000000000000000000000000000000000",
        ))
        .unwrap();
        assert_eq!(num, BigUint::from_u128(1 << 72).unwrap());

        // leading zeros are accepted
        let (_, num) = parse_unsigned_binary(ParseInput::new("0b0101")).unwrap();
        assert_eq!(num, BigUint::from_u128(4 + 1).unwrap());
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

        let (rest, num) = parse_unsigned_decimal(ParseInput::new(" 0a ")).unwrap();
        assert_eq!(num, BigUint::from_u128(0).unwrap());
        assert_eq!(rest, unsafe {
            LocatedSpan::new_from_raw_offset(2, 1, "a ", ())
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
    fn test_parse_signed_decimal() -> () {
        // at least 1 digit is required
        parse_signed_decimal(ParseInput::new("")).expect_err("");

        parse_signed_decimal(ParseInput::new("q0")).expect_err("");

        // whitespace accepted
        let (_, num) = parse_signed_decimal(ParseInput::new(" - 1")).unwrap();
        assert_eq!(num, BigInt::from_i128(-1).unwrap());

        // negative accepted
        let (_, num) = parse_signed_decimal(ParseInput::new("-0")).unwrap();
        assert_eq!(num, BigInt::from_i128(0).unwrap());

        let (_, num) = parse_signed_decimal(ParseInput::new("-1")).unwrap();
        assert_eq!(num, BigInt::from_i128(-1).unwrap());

        let (_, num) = parse_signed_decimal(ParseInput::new("0")).unwrap();
        assert_eq!(num, BigInt::from_i128(0).unwrap());
        let (_, num) = parse_signed_decimal(ParseInput::new("1")).unwrap();
        assert_eq!(num, BigInt::from_i128(1).unwrap());

        // fractions not accepted
        parse_signed_decimal(ParseInput::new(".1")).expect_err("");

        // Ensure that radix is 10
        parse_signed_decimal(ParseInput::new("a")).expect_err("");
        parse_signed_decimal(ParseInput::new("A")).expect_err("");

        let (_, num) = parse_signed_decimal(ParseInput::new("-36893488147419103232")).unwrap();
        let two_tothe_65 = BigInt::from_i128(-36893488147419103232).unwrap();
        assert_eq!(num, two_tothe_65);
    }

    #[test]
    fn test_parse_literal_type() -> () {
        // must start with s or u
        parse_bit_type_shorthand(ParseInput::new("")).expect_err("");
        parse_bit_type_shorthand(ParseInput::new("1")).expect_err("");
        parse_bit_type_shorthand(ParseInput::new("a")).expect_err("");

        // no space after {s,u}
        parse_bit_type_shorthand(ParseInput::new(" s 1 ")).expect_err("");
        parse_bit_type_shorthand(ParseInput::new(" u 1 ")).expect_err("");

        parse_bit_type_shorthand(ParseInput::new(" s1 ")).expect("");
        parse_bit_type_shorthand(ParseInput::new(" u1 ")).expect("");

        // TODO what to do about 0 width? error or ok?
        parse_bit_type_shorthand(ParseInput::new("s0")).expect("");
        parse_bit_type_shorthand(ParseInput::new("u0")).expect("");

        let (_, r) = parse_bit_type_shorthand(ParseInput::new("s1")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Signed,
                width: Usize(1)
            }
        );

        let (_, r) = parse_bit_type_shorthand(ParseInput::new("u1")).unwrap();
        assert_eq!(
            r.thing,
            RawBitType {
                signedness: Signedness::Unsigned,
                width: Usize(1)
            }
        );

        // 64 is the largest
        parse_bit_type_shorthand(ParseInput::new("u64")).expect("");
        parse_bit_type_shorthand(ParseInput::new("s64")).expect("");
        parse_bit_type_shorthand(ParseInput::new("u65")).expect_err("");
        parse_bit_type_shorthand(ParseInput::new("s65")).expect_err("");
    }
}
