// Provides an implementation of a scanner for the DSLX language.
// This is a work in progress and currently only supports a subset of valid tokens.
// TODO: I'm not sure how close this is to supporting UTF-8/Unicode. No promises yet.
use std::error::Error;
use std::vec::Vec;

pub mod token;

use token::Token;
use token::TokenKind;
use token::Keyword;

pub struct Scanner {
  // The input string.
  data: Vec<char>,

  // The current position in the file _after_ consuming the "peeked" character(s).
  cur_pos: usize,

  // Read-ahead.
  peek: Vec<char>,
}

impl Scanner {
  // Returns the next token from the input stream or an error.
  pub fn get_next_token(&mut self) -> Result<token::Token, Box<dyn Error>> {
    self.consume_whitespace();

    let c = self.peek_char()?;
    let res = match c {
      '(' => { self.pop_char_or_die(); Token { kind: token::TokenKind::OParen } },
      ')' => { self.pop_char_or_die(); Token { kind: token::TokenKind::CParen } },
      '{' => { self.pop_char_or_die(); Token { kind: token::TokenKind::OBrace } },
      '}' => { self.pop_char_or_die(); Token { kind: token::TokenKind::CBrace } },
      ':' => { self.pop_char_or_die(); Token { kind: token::TokenKind::Colon } },
      '+' => { self.pop_char_or_die(); Token { kind: token::TokenKind::Plus } },
      '-' => {
        self.pop_char_or_die();
        let c = self.peek_char();
        if c.is_ok() && c.unwrap() == '>' {
          self.pop_char_or_die();
          Token  { kind: token::TokenKind::Arrow }
        } else {
          Token  { kind: token::TokenKind::Minus }
        }
      },
      'i' | 'u' => { self.pop_char_or_die(); self.read_int_type(c == 'i')? },
      '_' | 'a'..='z' | 'A'..='Z' => self.read_string()?,
      '0'..='9' => Token { kind: token::TokenKind::Number(self.read_number()?) },
      _ => Err(format!("Unexpected char: {}", c))?
    };
    
    Ok(res)
  }

  fn pop_char_or_die(&mut self) -> char {
    self.pop_char().unwrap()
  }

  fn pop_char(&mut self) -> Result<char, Box<dyn Error>> {
    let ret = self.peek_char()?;
    self.peek.pop();
    Ok(ret)
  }

  // We'll need to support multiple peeks at some point, probably...but not yet.
  fn peek_char(&mut self) -> Result<char, Box<dyn Error>> {
    if !self.peek.is_empty() {
      return Ok(*self.peek.last().unwrap());
    }

    if self.cur_pos == self.data.len() {
      return Err("EOF reached.")?
    }

    let ret = self.data[self.cur_pos];
    self.peek.push(ret);
    self.cur_pos += 1;
    Ok(ret)
  }

  fn read_number(&mut self) -> Result<i64, Box<dyn Error>> {
    fn read_digits(scanner: &mut Scanner, num_str: &mut String, radix: u32)
        -> Result<i64, Box<dyn Error>> {
      let mut c_or = scanner.peek_char();
      while c_or.is_ok() {
        let c = c_or.unwrap();
        if c.is_whitespace() || !c.is_digit(radix) {
          break;
        }

        num_str.push(c);
        scanner.pop_char_or_die();
        c_or = scanner.peek_char();
      }

      return Ok(i64::from_str_radix(num_str, radix)?);
    }

    // Or die is safe; we know we have one char, at least.
    let mut c = self.pop_char_or_die();
    let mut num_str = String::new();
    num_str.push(c);

    if c != '0' {
      return read_digits(self, &mut num_str, /*radix=*/10);
    }

    c = self.peek_char()?;
    if c == 'x' {
      num_str.push(c);
      return read_digits(self, &mut num_str, /*radix=*/16);
    }

    // Handles the "0" case, too.
    return read_digits(self, &mut num_str, /*radix=*/8);
  }

  // Error handling can/should be improved (add a clear message).
  fn read_int_type(&mut self, signed: bool) -> Result<token::Token, Box<dyn Error>> {
    let bit_count = usize::try_from(self.read_number()?)?;
    return Ok(Token { kind: if signed { crate::TokenKind::SIntType(bit_count) }
      else { crate::TokenKind::UIntType(bit_count) } })
  }

  fn read_string(&mut self) -> Result<token::Token, Box<dyn Error>> {
    fn is_string_char(x: char) -> bool {
      // This is only for chars aside from the first, which is already guaranteed to be valid.
      x.is_alphanumeric() || x == '_'
    }

    let mut cur_str = String::new();
    cur_str.push(self.pop_char()?);

    let mut cur_char = self.peek_char();
    while cur_char.is_ok() && is_string_char(cur_char.unwrap()) {
      cur_str.push(self.pop_char()?);
      cur_char = self.peek_char();
    }

    // Is there a better Rusty way to iterate over the possible keywords?
    // Match i32, e.g.
    let token_kind = match cur_str.as_str() {
      "fn" => token::TokenKind::Keyword(token::Keyword::_Fn),
      _ => token::TokenKind::Identifier(cur_str),
    };

    Ok(Token { kind: token_kind })
  }

  fn consume_whitespace(&mut self) -> Option<Box<dyn Error>> {
    while self.peek_char().ok()?.is_whitespace() {
      self.pop_char().ok()?;
    }
    None
  }
}

pub fn create_scanner(input: &str) -> Result<Scanner, Box<dyn Error>> {
  let scanner = Scanner {
    data: input.chars().collect(),
    cur_pos: 0,
    peek: Vec::new(),
  };
  Ok(scanner)
}

#[cfg(test)]
mod tests {

use crate::create_scanner;

// "Smoke test": can scan a near-trivial function.
#[test]
fn simple_scan() {
  const INPUT : &str = "fn add1(x: u32) -> u32 { x + u32:1 }";

  fn check_next_token(scanner: &mut crate::Scanner, kind: crate::TokenKind) {
    let token_or = scanner.get_next_token();
    assert!(token_or.is_ok(), "Ran out of tokens: {}", token_or.unwrap_err());
    let token = token_or.unwrap();
    assert_eq!(token.kind, kind);
  }

  let scanner_or = create_scanner(INPUT);
  assert!(scanner_or.is_ok());

  let mut scanner = scanner_or.unwrap();
  check_next_token(&mut scanner, crate::TokenKind::Keyword(crate::Keyword::_Fn));
  check_next_token(&mut scanner, crate::TokenKind::Identifier("add1".to_string()));
  check_next_token(&mut scanner, crate::TokenKind::OParen);
  check_next_token(&mut scanner, crate::TokenKind::Identifier("x".to_string()));
  check_next_token(&mut scanner, crate::TokenKind::Colon);
  check_next_token(&mut scanner, crate::TokenKind::UIntType(32));
  check_next_token(&mut scanner, crate::TokenKind::CParen);
  check_next_token(&mut scanner, crate::TokenKind::Arrow);
  check_next_token(&mut scanner, crate::TokenKind::UIntType(32));
  check_next_token(&mut scanner, crate::TokenKind::OBrace);
  check_next_token(&mut scanner, crate::TokenKind::Identifier("x".to_string()));
  check_next_token(&mut scanner, crate::TokenKind::Plus);
  check_next_token(&mut scanner, crate::TokenKind::UIntType(32));
  check_next_token(&mut scanner, crate::TokenKind::Colon);
  check_next_token(&mut scanner, crate::TokenKind::Number(1));
  check_next_token(&mut scanner, crate::TokenKind::CBrace);
}

}
