//! Defines data types and related for the Abstract Syntax Tree.
//!
//! Naming convention: RawFoo is the Foo not including the `Span`. Foo will include the `Span`.

use nonempty::NonEmpty;
use num_bigint::{BigInt, BigUint};
use std::cmp::Ordering;

/// Defines the types in the AST for DSLX.
pub type ParseInput<'a> = nom_locate::LocatedSpan<&'a str>;

// TODO: Move pos/span to own file.
/// A distinct position in the input: offset from input start, line number, and column within line.
#[derive(Debug, PartialEq, Clone)]
pub struct Pos {
    /// Offset from the start of overall parse input in bytes/characters.
    /// Unicode is not _currently_ supported.
    pub input_offset: usize,
    /// 1-indexed line number from parse input.
    pub line: usize,
    /// 1-indexed column number in the current line.
    pub column: usize,
}

impl Pos {
    pub fn new(x: ParseInput) -> Pos {
        Pos {
            input_offset: x.location_offset(),
            line: x.location_line() as usize,
            column: x.get_column(),
        }
    }
}

impl From<(usize, usize, usize)> for Pos {
    fn from(item: (usize, usize, usize)) -> Self {
        Pos {
            input_offset: item.0,
            line: item.1,
            column: item.2,
        }
    }
}

/// Identifies a range of the parser input. Enables us to track where in the source text some
/// element originated, so that we can report errors to the user.
#[derive(Debug, PartialEq, Clone)]
pub struct Span {
    pub start: Pos,
    /// Not inclusive, i.e., this represents the first position after the end of the spanned region.
    // TODO: Make this inclusive (not completely trivial, since it might
    // require "backing up" a line & newline handling, ... Not horribly hard, either, just adequate
    // for its own change).
    pub end: Pos,
}

impl Span {
    pub fn new(start: ParseInput, end: ParseInput) -> Span {
        Span {
            start: Pos::new(start),
            end: Pos::new(end),
        }
    }
}

impl From<((usize, usize, usize), (usize, usize, usize))> for Span {
    fn from(item: ((usize, usize, usize), (usize, usize, usize))) -> Self {
        Span {
            start: Pos::from(item.0),
            end: Pos::from(item.1),
        }
    }
}

/// Represents a name of an entity, such as a type, variable, function, etc.
#[derive(Debug, PartialEq, Clone)]
pub struct RawIdentifier(pub String);

pub type Identifier = Spanned<RawIdentifier>;

impl<'a> From<ParseInput<'a>> for RawIdentifier {
    fn from(span: ParseInput<'a>) -> Self {
        RawIdentifier((*span.fragment()).to_owned())
    }
}

/// Introduces a variable with its name and type, e.g., `foo : MyType`. E.g. used in function
/// type signature parameter lists, `let` bindings, etc.
#[derive(Debug, PartialEq, Clone)]
pub struct RawBindingDecl {
    pub name: Identifier,
    pub typ: Identifier,
}

pub type BindingDecl = Spanned<RawBindingDecl>;
pub type BindingDeclList = Spanned<Vec<BindingDecl>>;

impl From<(Identifier, Identifier)> for RawBindingDecl {
    fn from((name, typ): (Identifier, Identifier)) -> Self {
        RawBindingDecl { name, typ }
    }
}

/// A function signature, e.g: `fn foo(x:u32) -> u32`.
#[derive(Debug, PartialEq)]
pub struct RawFunctionSignature {
    pub name: Identifier,
    pub parameters: BindingDeclList,
    pub result_type: Identifier,
}

pub type FunctionSignature = Spanned<RawFunctionSignature>;

impl From<(Identifier, BindingDeclList, Identifier)> for RawFunctionSignature {
    fn from((name, parameters, result_type): (Identifier, BindingDeclList, Identifier)) -> Self {
        RawFunctionSignature {
            name,
            parameters,
            result_type,
        }
    }
}

/// Indicates a signed or unsigned integer.
#[derive(Debug, PartialEq, Clone)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// Values that turn into array dimensions (AKA bit widths) all become this.
///
/// See https://github.com/google/xls/issues/450 to understand why we have this type, instead
/// of just using `u32` to store bit widths.
#[derive(Debug, PartialEq, Clone)]
pub struct Usize(pub u32);

/// The "variable length bit type" is "The most fundamental type in DSLX". It has a width (AKA
/// length) and signedness. E.g.:
///
/// `u16` is 16 bits and unsigned
///
/// `s8` is 8 bits and signed
///
/// See <https://google.github.io/xls/dslx_reference/#bit-type>
#[derive(Debug, PartialEq, Clone)]
pub struct RawBitType {
    /// A bit type is differentiated on if it's signed or unsigned.
    pub signedness: Signedness,

    /// Width, in bits
    pub width: Usize,
}

pub type BitType = Spanned<RawBitType>;

impl From<(Signedness, u32)> for RawBitType {
    fn from((signedness, width): (Signedness, u32)) -> Self {
        RawBitType {
            signedness,
            width: Usize(width),
        }
    }
}

/// A (big) integer, tagged Unsigned or Signed.
#[derive(Debug, PartialEq, Clone)]
pub enum RawInteger {
    Unsigned(BigUint),
    Signed(BigInt),
}

impl From<BigUint> for RawInteger {
    fn from(x: BigUint) -> Self {
        RawInteger::Unsigned(x)
    }
}

impl From<BigInt> for RawInteger {
    fn from(x: BigInt) -> Self {
        RawInteger::Signed(x)
    }
}

pub type Integer = Spanned<RawInteger>;

/// A literal, e.g. `s4:0b1001`.
#[derive(Debug, PartialEq, Clone)]
pub struct RawLiteral {
    pub value: Integer,
    pub bit_type: BitType,
}

pub type Literal = Spanned<RawLiteral>;

/// Operators for unary expressions
///
/// see <https://google.github.io/xls/dslx_reference/#unary-expressions>
#[derive(Debug, PartialEq, Clone)]
pub enum RawUnaryOperator {
    /// `-`, computes the two's complement negation
    Negate,
    /// `!`, AKA bit-wise not
    Invert,
}

pub type UnaryOperator = Spanned<RawUnaryOperator>;

/// Operators for binary expressions.
///
/// see <https://google.github.io/xls/dslx_reference/#binary-expressions>
/// <https://google.github.io/xls/dslx_reference/#shift-expressions>
/// <https://google.github.io/xls/dslx_reference/#comparison-expressions>
/// <https://google.github.io/xls/dslx_reference/#concat-expression>
///
/// They are ordered from highest precedence to lowest, and grouped when same precedence.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RawBinaryOperator {
    /// `*`, multiply
    Multiply,

    /// `+`, add
    Add,
    /// `-`, subtract
    Subtract,
    /// `++` bitwise concatenation
    Concatenate,

    /// `>>`, shift right (both logical and arithmetic, depending on context)
    ShiftRight,
    /// `<<`, shift left
    ShiftLeft,

    /// `&`, bit-wise and
    BitwiseAnd,

    /// `^`, bit-wise xor
    BitwiseXor,

    /// `|`, bit-wise or
    BitwiseOr,

    /// `==` equal
    Equal,
    /// `!=` not equal
    NotEqual,
    /// `>=` greater or equal
    GreaterOrEqual,
    /// `>`  greater
    Greater,
    /// `<=` less or equal
    LessOrEqual,
    /// `<`  less
    Less,

    /// `&&`, boolean and
    BooleanAnd,

    /// `||`, boolean or
    BooleanOr,
}

pub type BinaryOperator = Spanned<RawBinaryOperator>;

impl RawBinaryOperator {
    /// Returns true for the 6 comparison operators (== != < > <= >=), otherwise false.
    fn is_comparison(&self) -> bool {
        match self {
            RawBinaryOperator::Multiply => false,
            RawBinaryOperator::Add => false,
            RawBinaryOperator::Subtract => false,
            RawBinaryOperator::Concatenate => false,
            RawBinaryOperator::ShiftRight => false,
            RawBinaryOperator::ShiftLeft => false,
            RawBinaryOperator::BitwiseAnd => false,
            RawBinaryOperator::BitwiseXor => false,
            RawBinaryOperator::BitwiseOr => false,
            RawBinaryOperator::Equal => true,
            RawBinaryOperator::NotEqual => true,
            RawBinaryOperator::GreaterOrEqual => true,
            RawBinaryOperator::Greater => true,
            RawBinaryOperator::LessOrEqual => true,
            RawBinaryOperator::Less => true,
            RawBinaryOperator::BooleanAnd => false,
            RawBinaryOperator::BooleanOr => false,
        }
    }
}

impl PartialOrd for RawBinaryOperator {
    /// Implements DSLX binary operator precedence. I.e. `x Some(Greater) y` means that x is
    /// higher in the precedence table than y. `x None y` means they're the same precedence
    /// (but they're not necessarily equal) see
    /// <https://google.github.io/xls/dslx_reference/#operator-precedence>
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if PartialEq::eq(self, other) {
            Some(Ordering::Equal)
        } else {
            match (self, other) {
                (Self::Multiply, _) => Some(Ordering::Greater),
                (_, Self::Multiply) => Some(Ordering::Less),

                // TODO verify:
                // If I'm reading
                // https://github.com/google/xls/blob/main/xls/dslx/frontend/parser.h#L360C53-L360C74
                // correctly, ++ is same precedence as + and -
                (Self::Add, Self::Subtract) => None,
                (Self::Add, Self::Concatenate) => None,
                (Self::Subtract, Self::Add) => None,
                (Self::Subtract, Self::Concatenate) => None,
                (Self::Concatenate, Self::Add) => None,
                (Self::Concatenate, Self::Subtract) => None,
                (Self::Add, _) => Some(Ordering::Greater),
                (Self::Subtract, _) => Some(Ordering::Greater),
                (Self::Concatenate, _) => Some(Ordering::Greater),
                (_, Self::Add) => Some(Ordering::Less),
                (_, Self::Subtract) => Some(Ordering::Less),
                (_, Self::Concatenate) => Some(Ordering::Less),

                // shift
                (Self::ShiftRight, Self::ShiftLeft) => None,
                (Self::ShiftLeft, Self::ShiftRight) => None,
                (Self::ShiftRight, _) => Some(Ordering::Greater),
                (Self::ShiftLeft, _) => Some(Ordering::Greater),
                (_, Self::ShiftRight) => Some(Ordering::Less),
                (_, Self::ShiftLeft) => Some(Ordering::Less),

                // bitwise are different precedence from each other
                (Self::BitwiseAnd, _) => Some(Ordering::Greater),
                (_, Self::BitwiseAnd) => Some(Ordering::Less),
                (Self::BitwiseXor, _) => Some(Ordering::Greater),
                (_, Self::BitwiseXor) => Some(Ordering::Less),
                (Self::BitwiseOr, _) => Some(Ordering::Greater),
                (_, Self::BitwiseOr) => Some(Ordering::Less),

                (Self::Equal, other) if other.is_comparison() => None,
                (Self::NotEqual, other) if other.is_comparison() => None,
                (Self::GreaterOrEqual, other) if other.is_comparison() => None,
                (Self::Greater, other) if other.is_comparison() => None,
                (Self::LessOrEqual, other) if other.is_comparison() => None,
                (Self::Less, other) if other.is_comparison() => None,
                (Self::Equal, _) => Some(Ordering::Greater),
                (Self::NotEqual, _) => Some(Ordering::Greater),
                (Self::GreaterOrEqual, _) => Some(Ordering::Greater),
                (Self::Greater, _) => Some(Ordering::Greater),
                (Self::LessOrEqual, _) => Some(Ordering::Greater),
                (Self::Less, _) => Some(Ordering::Greater),
                (_, Self::Equal) => Some(Ordering::Less),
                (_, Self::NotEqual) => Some(Ordering::Less),
                (_, Self::GreaterOrEqual) => Some(Ordering::Less),
                (_, Self::Greater) => Some(Ordering::Less),
                (_, Self::LessOrEqual) => Some(Ordering::Less),
                (_, Self::Less) => Some(Ordering::Less),

                (Self::BooleanAnd, _) => Some(Ordering::Greater),
                (_, Self::BooleanAnd) => Some(Ordering::Less),

                // We avoid a wildcard match so that we update this logic when new cases are added
                //
                // Required to satisfy the exhaustiveness checker
                (Self::BooleanOr, Self::BooleanOr) => {
                    panic!("logically impossible because of PartialEq::eq(self, other) above")
                }
            }
        }
    }
}

/// *Part* of a let binding: the variable declaration and value it is bound to. What is missing
/// is the expression in which the bound name is in-scope.
#[derive(Debug, PartialEq, Clone)]
pub struct RawLetBinding {
    pub variable_declaration: BindingDecl,
    pub value: Box<Expression>,
}
pub type LetBinding = Spanned<RawLetBinding>;

// This struct exists to ensure that `From<Expression> for RawExpression` does not exist
// (because instead we have `From<ParenthesizedExpression> for RawExpression`). The former was
// bug prone: I was accidentally and unknowingly calling `from(Expression) -> RawExpression`.
// Inside the `from` we will discard the ParenthesizedExpression 'wrapper'.
#[derive(Debug, PartialEq, Clone)]
pub struct ParenthesizedExpression(pub Expression);

/// An expression (i.e. a thing that can be evaluated), e.g. `s1:1 + s1:0`.
#[derive(Debug, PartialEq, Clone)]
pub enum RawExpression {
    /// A literal, e.g. `s4:0b1001`
    Literal(Literal),

    /// A name bound to a value (e.g. by a previous `let` expression, or a function argument).
    Binding(Identifier),

    /// An expression that's surrounded by an open and close parentheses. The expression inside
    /// the parentheses will be evaluated with the highest precedence.
    Parenthesized(Box<Expression>),

    /// a unary expression, e.g. `!s4:0b1001`
    Unary(UnaryOperator, Box<Expression>),

    /// a binary expression, e.g. `s1:1 + s1:0`
    Binary(Box<Expression>, BinaryOperator, Box<Expression>),

    /// 1 or more let expressions (i.e. the vector may be empty).
    ///
    /// Every binding is in scope in the bindings that come after it in the vector (i.e. a
    /// binding is lexically scoped). The final expression, if present (and we expect it to
    /// exist most of the time, otherwise, why bother with an if expression), can use all the
    /// bindings. When absent, the value of the let expression is `()`.
    Let(NonEmpty<LetBinding>, Option<Box<Expression>>),
}

impl From<Literal> for RawExpression {
    fn from(x: Literal) -> Self {
        RawExpression::Literal(x)
    }
}

impl From<Identifier> for RawExpression {
    fn from(x: Identifier) -> Self {
        RawExpression::Binding(x)
    }
}

impl From<ParenthesizedExpression> for RawExpression {
    fn from(ParenthesizedExpression(x): ParenthesizedExpression) -> Self {
        RawExpression::Parenthesized(Box::new(x))
    }
}

impl From<(UnaryOperator, Expression)> for RawExpression {
    fn from((op, expr): (UnaryOperator, Expression)) -> Self {
        RawExpression::Unary(op, Box::new(expr))
    }
}

impl From<(Expression, BinaryOperator, Expression)> for RawExpression {
    fn from((lhs, op, rhs): (Expression, BinaryOperator, Expression)) -> Self {
        RawExpression::Binary(Box::new(lhs), op, Box::new(rhs))
    }
}

impl From<(NonEmpty<LetBinding>, Option<Expression>)> for RawExpression {
    fn from((bindings, using_expr): (NonEmpty<LetBinding>, Option<Expression>)) -> Self {
        RawExpression::Let(bindings, using_expr.map(Box::new))
    }
}

pub type Expression = Spanned<RawExpression>;

/// A parsed thing (e.g. `Identifier`) and the corresponding Span in the source text.
#[derive(Debug, PartialEq, Clone)]
pub struct Spanned<Thing> {
    pub span: Span,
    pub thing: Thing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn test_raw_binary_operator_partial_ord() -> () {
        // Asserts: this > that AND that < this
        fn this_greater_than_that(this: RawBinaryOperator, that: RawBinaryOperator) -> () {
            assert_eq!(this.partial_cmp(&that), Some(Ordering::Greater));
            assert_eq!(this > that, true);
            assert_eq!(this >= that, true);
            assert_eq!(that.partial_cmp(&this), Some(Ordering::Less));
            assert_eq!(that < this, true);
            assert_eq!(that <= this, true);

            assert_eq!(this == that, false);
            assert_eq!(that == this, false);

            assert_eq!(this != that, true);
            assert_eq!(that != this, true);
        }

        // Asserts: this and that are unordered and not equal
        fn same_precedence(this: RawBinaryOperator, that: RawBinaryOperator) -> () {
            assert_eq!(this.partial_cmp(&that), None);
            assert_eq!(this > that, false);
            assert_eq!(this < that, false);
            assert_eq!(this <= that, false);
            assert_eq!(this >= that, false);
            assert_eq!(this == that, false);

            assert_eq!(that.partial_cmp(&this), None);
            assert_eq!(that > this, false);
            assert_eq!(that < this, false);
            assert_eq!(that <= this, false);
            assert_eq!(that >= this, false);
            assert_eq!(that == this, false);

            assert_eq!(this != that, true);
            assert_eq!(that != this, true);
        }

        // partial_cmp says reflexive is equal
        fn is_reflexive(this: RawBinaryOperator) -> () {
            assert_eq!(this.partial_cmp(&this), Some(Ordering::Equal));
            assert_eq!(this == this, true);
            assert_eq!(this != this, false);
        }

        is_reflexive(RawBinaryOperator::Multiply);
        is_reflexive(RawBinaryOperator::Add);
        is_reflexive(RawBinaryOperator::Subtract);
        is_reflexive(RawBinaryOperator::Concatenate);
        is_reflexive(RawBinaryOperator::ShiftRight);
        is_reflexive(RawBinaryOperator::ShiftLeft);
        is_reflexive(RawBinaryOperator::BitwiseAnd);
        is_reflexive(RawBinaryOperator::BitwiseXor);
        is_reflexive(RawBinaryOperator::BitwiseOr);
        is_reflexive(RawBinaryOperator::Equal);
        is_reflexive(RawBinaryOperator::NotEqual);
        is_reflexive(RawBinaryOperator::GreaterOrEqual);
        is_reflexive(RawBinaryOperator::Greater);
        is_reflexive(RawBinaryOperator::LessOrEqual);
        is_reflexive(RawBinaryOperator::Less);
        is_reflexive(RawBinaryOperator::BooleanAnd);
        is_reflexive(RawBinaryOperator::BooleanOr);

        // Multiply highest precedence
        let it = vec![
            RawBinaryOperator::Add,
            RawBinaryOperator::Subtract,
            RawBinaryOperator::Concatenate,
            RawBinaryOperator::ShiftRight,
            RawBinaryOperator::ShiftLeft,
            RawBinaryOperator::BitwiseAnd,
            RawBinaryOperator::BitwiseXor,
            RawBinaryOperator::BitwiseOr,
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::Multiply, op);
        }

        // weak arithmetic same prec
        for ops in vec![
            RawBinaryOperator::Add,
            RawBinaryOperator::Subtract,
            RawBinaryOperator::Concatenate,
        ]
        .into_iter()
        .combinations(2)
        {
            same_precedence(ops[0], ops[1]);
        }

        // weak arithmetic next-highest precedence
        let it = vec![
            RawBinaryOperator::ShiftRight,
            RawBinaryOperator::ShiftLeft,
            RawBinaryOperator::BitwiseAnd,
            RawBinaryOperator::BitwiseXor,
            RawBinaryOperator::BitwiseOr,
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::Add, op);
            this_greater_than_that(RawBinaryOperator::Subtract, op);
            this_greater_than_that(RawBinaryOperator::Concatenate, op);
        }

        // shift same precedence
        same_precedence(RawBinaryOperator::ShiftLeft, RawBinaryOperator::ShiftRight);

        // shift next-highest precedence
        let it = vec![
            RawBinaryOperator::BitwiseAnd,
            RawBinaryOperator::BitwiseXor,
            RawBinaryOperator::BitwiseOr,
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::ShiftRight, op);
            this_greater_than_that(RawBinaryOperator::ShiftLeft, op);
        }

        // bitwise AND next-highest precedence
        let it = vec![
            RawBinaryOperator::BitwiseXor,
            RawBinaryOperator::BitwiseOr,
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::BitwiseAnd, op);
        }

        // bitwise XOR next-highest precedence
        let it = vec![
            RawBinaryOperator::BitwiseOr,
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::BitwiseXor, op);
        }

        // bitwise OR next-highest precedence
        let it = vec![
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
            RawBinaryOperator::BooleanAnd,
            RawBinaryOperator::BooleanOr,
        ]
        .into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::BitwiseOr, op);
        }

        // comparison
        for ops in vec![
            RawBinaryOperator::Equal,
            RawBinaryOperator::NotEqual,
            RawBinaryOperator::GreaterOrEqual,
            RawBinaryOperator::Greater,
            RawBinaryOperator::LessOrEqual,
            RawBinaryOperator::Less,
        ]
        .into_iter()
        .combinations(2)
        {
            same_precedence(ops[0], ops[1]);
        }
        // comparison next-highest precedence
        let it = vec![RawBinaryOperator::BooleanAnd, RawBinaryOperator::BooleanOr].into_iter();
        for op in it {
            this_greater_than_that(RawBinaryOperator::Equal, op);
            this_greater_than_that(RawBinaryOperator::NotEqual, op);
            this_greater_than_that(RawBinaryOperator::GreaterOrEqual, op);
            this_greater_than_that(RawBinaryOperator::Greater, op);
            this_greater_than_that(RawBinaryOperator::LessOrEqual, op);
            this_greater_than_that(RawBinaryOperator::Less, op);
        }

        // last two
        this_greater_than_that(RawBinaryOperator::BooleanAnd, RawBinaryOperator::BooleanOr);
    }
}
