// Contains descriptions for the valid tokens in DSLX.
// Work in progress - only currently supports a subset of the language's tokens.

// The "_" is added to disambiguate terms that are valid Rust keywords from those used
// in DSLX parsing, since the two languages have a lot in common.
#[derive(Debug, PartialEq)]
pub enum Keyword {
  _Fn,
  _For,
}

#[derive(Debug, PartialEq)]
pub enum TokenKind {
  Keyword(Keyword),
  Identifier(String),
  SIntType(usize),
  UIntType(usize),
  Number(i64),
  OParen,
  CParen,
  OBrace,
  CBrace,
  OBrack,
  CBrack,
  Colon,
  Plus,
  Minus,
  Arrow,
  FatArrow,
}

// Currently a placeholder, but will grow to include things like position and span.
#[derive(Debug)]
pub struct Token {
  pub kind: TokenKind,
}
