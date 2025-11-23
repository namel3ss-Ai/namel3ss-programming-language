# Namel3ss (N3) Formal Grammar Specification

**Version:** 1.0.0  
**Date:** November 21, 2025  
**Status:** Canonical

This document defines the complete formal grammar for the Namel3ss programming language using Extended Backus-Naur Form (EBNF).

---

## 1. Lexical Structure

### 1.1 Whitespace and Comments

```ebnf
(* Whitespace *)
WHITESPACE      = ( " " | "\t" | "\r" | "\n" )+ ;

(* Comments *)
LINE_COMMENT    = ( "#" | "//" ) , { ANY_CHAR - "\n" } , "\n" ;
BLOCK_COMMENT   = "/*" , { ANY_CHAR } , "*/" ;

COMMENT         = LINE_COMMENT | BLOCK_COMMENT ;
```

### 1.2 Identifiers and Keywords

```ebnf
(* Identifiers *)
IDENTIFIER      = LETTER , { LETTER | DIGIT | "_" } ;
LETTER          = "a".."z" | "A".."Z" ;
DIGIT           = "0".."9" ;

(* Reserved Keywords *)
KEYWORD         = "app" | "page" | "llm" | "agent" | "prompt" | "chain" 
                | "rag_pipeline" | "index" | "dataset" | "memory" | "function"
                | "fn" | "tool" | "connector" | "template" | "model" | "training"
                | "policy" | "graph" | "knowledge" | "query" | "frame" | "theme"
                | "import" | "module" | "language_version"
                | "if" | "else" | "for" | "while" | "match" | "case"
                | "true" | "false" | "null" | "env" | "let" | "in"
                | "show" | "log" | "debug" | "info" | "warn" | "error"
                | "filter" | "map" | "transform" ;
```

### 1.3 Literals

```ebnf
(* String Literals *)
STRING_LITERAL  = DOUBLE_QUOTED | SINGLE_QUOTED | TRIPLE_QUOTED ;
DOUBLE_QUOTED   = '"' , { STRING_CHAR | ESCAPED_CHAR } , '"' ;
SINGLE_QUOTED   = "'" , { STRING_CHAR | ESCAPED_CHAR } , "'" ;
TRIPLE_QUOTED   = '"""' , { ANY_CHAR } , '"""' ;

ESCAPED_CHAR    = "\\" , ( "n" | "t" | "r" | "\\" | '"' | "'" | "{" | "}" ) ;
STRING_CHAR     = ANY_CHAR - ( '"' | "'" | "\\" | "\n" ) ;

(* Numeric Literals *)
NUMBER          = INTEGER | FLOAT ;
INTEGER         = [ "-" ] , DIGIT+ ;
FLOAT           = [ "-" ] , DIGIT+ , "." , DIGIT+ [ EXPONENT ] ;
EXPONENT        = ( "e" | "E" ) , [ "+" | "-" ] , DIGIT+ ;

(* Boolean and Null *)
BOOLEAN         = "true" | "false" ;
NULL            = "null" ;
```

### 1.4 Operators and Punctuation

```ebnf
(* Operators *)
OPERATOR        = "+" | "-" | "*" | "/" | "%" | "**"
                | "==" | "!=" | "<" | ">" | "<=" | ">="
                | "&&" | "||" | "!"
                | "=>" | "->" | "|" | "&" ;

(* Punctuation *)
PUNCTUATION     = "{" | "}" | "[" | "]" | "(" | ")" 
                | ":" | ";" | "," | "." | "..." ;
```

---

## 2. Program Structure

### 2.1 Module

```ebnf
(* Top-level program structure *)
Module          = [ ModuleDirectives ] , { TopLevelDecl } ;

ModuleDirectives = { ModuleDirective } ;
ModuleDirective = ModuleDecl | ImportDecl | LanguageVersion ;

ModuleDecl      = "module" , STRING_LITERAL , "\n" ;
ImportDecl      = "import" , ImportPath , [ "as" , IDENTIFIER ] , "\n" ;
ImportPath      = IDENTIFIER , { "." , IDENTIFIER } ;
LanguageVersion = "language_version" , ":" , STRING_LITERAL , "\n" ;
```

### 2.2 Top-Level Declarations

```ebnf
TopLevelDecl    = AppDecl
                | PageDecl
                | LLMDecl
                | AgentDecl
                | PromptDecl
                | ChainDecl
                | RAGPipelineDecl
                | IndexDecl
                | DatasetDecl
                | MemoryDecl
                | FunctionDecl
                | ToolDecl
                | ConnectorDecl
                | TemplateDecl
                | ModelDecl
                | TrainingDecl
                | PolicyDecl
                | GraphDecl
                | KnowledgeDecl
                | FrameDecl
                | ThemeDecl ;
```

---

## 3. Core Declarations

### 3.1 Application Declaration

```ebnf
(* Canonical Syntax *)
AppDecl         = "app" , QuotedName , [ AppConnections ] , Block ;

QuotedName      = STRING_LITERAL ;
AppConnections  = "connects" , "to" , ConnectionList ;
ConnectionList  = Connection , { "," , Connection } ;
Connection      = DatabaseConnection | ServiceConnection ;
DatabaseConnection = ( "postgres" | "mysql" | "mongodb" ) , QuotedName ;
ServiceConnection  = IDENTIFIER , QuotedName ;

Block           = "{" , "\n" , { BlockStatement } , "}" ;
BlockStatement  = KeyValuePair | NestedDecl ;
KeyValuePair    = IDENTIFIER , ":" , Value , "\n" ;
```

### 3.2 Page Declaration

```ebnf
PageDecl        = "page" , QuotedName , "at" , STRING_LITERAL , Block ;

PageBlock       = "{" , "\n" , { PageStatement } , "}" ;
PageStatement   = ShowStatement | LogStatement | ControlFlow | Expression ;

ShowStatement   = "show" , ComponentType , [ ComponentConfig ] , "\n" ;
ComponentType   = "text" | "table" | "chart" | "form" | "button" 
                | "input" | "select" | "image" | "video" ;
ComponentConfig = ":" , ( InlineConfig | Block ) ;

LogStatement    = "log" , [ LogLevel ] , STRING_LITERAL , "\n" ;
LogLevel        = "debug" | "info" | "warn" | "error" ;
```

### 3.3 LLM Declaration

```ebnf
(* Canonical unified syntax *)
LLMDecl         = "llm" , QuotedName , Block ;

LLMBlock        = "{" , "\n" , LLMConfig , "}" ;
LLMConfig       = { LLMConfigItem } ;
LLMConfigItem   = ( "provider" | "model" | "api_key" | "temperature" 
                  | "max_tokens" | "top_p" | "frequency_penalty" 
                  | "presence_penalty" ) , ":" , Value , "\n" ;
```

### 3.4 Agent Declaration

```ebnf
AgentDecl       = "agent" , QuotedName , Block ;

AgentBlock      = "{" , "\n" , { AgentConfigItem } , "}" ;
AgentConfigItem = ( "llm" | "system_prompt" | "tools" | "memory" 
                  | "max_iterations" | "temperature" ) , ":" , Value , "\n" ;
```

### 3.5 Prompt Declaration

```ebnf
PromptDecl      = "prompt" , QuotedName , Block ;

PromptBlock     = "{" , "\n" , 
                  [ InputSchema ] ,
                  [ OutputSchema ] ,
                  TemplateSection ,
                  "}" ;

InputSchema     = "input" , ":" , SchemaDefinition , "\n" ;
OutputSchema    = "output" , ":" , SchemaDefinition , "\n" ;
TemplateSection = "template" , ":" , ( STRING_LITERAL | TripleQuotedString ) , "\n" ;

SchemaDefinition = ArraySchema | ObjectSchema ;
ArraySchema     = "[" , "\n" , { SchemaField } , "]" ;
SchemaField     = "-" , IDENTIFIER , ":" , TypeAnnotation , [ "(" , Constraint , ")" ] , "\n" ;
TypeAnnotation  = "text" | "number" | "boolean" | "array" | "object" ;
Constraint      = "required" | "optional" ;
```

### 3.6 Chain Declaration

```ebnf
ChainDecl       = "chain" , QuotedName , Block ;

ChainBlock      = "{" , "\n" , ChainSteps , "}" ;
ChainSteps      = StepDefinition , { StepConnector , StepDefinition } ;
StepDefinition  = IDENTIFIER | FunctionCall | LambdaExpr ;
StepConnector   = "->" | "|" ;  (* -> for sequential, | for parallel *)
```

### 3.7 RAG Pipeline Declaration

```ebnf
RAGPipelineDecl = "rag_pipeline" , QuotedName , Block ;

RAGBlock        = "{" , "\n" , { RAGConfigItem } , "}" ;
RAGConfigItem   = ( "query_encoder" | "index" | "top_k" | "distance_metric" 
                  | "retriever" | "reranker" | "chunker" ) , ":" , Value , "\n" ;
```

### 3.8 Index Declaration

```ebnf
IndexDecl       = "index" , QuotedName , Block ;

IndexBlock      = "{" , "\n" , { IndexConfigItem } , "}" ;
IndexConfigItem = ( "source_dataset" | "embedding_model" | "chunk_size" 
                  | "overlap" | "backend" | "table_name" | "dimensions" ) , ":" , Value , "\n" ;
```

### 3.9 Memory Declaration

```ebnf
MemoryDecl      = "memory" , QuotedName , Block ;

MemoryBlock     = "{" , "\n" , { MemoryConfigItem } , "}" ;
MemoryConfigItem = ( "scope" | "kind" | "max_items" | "max_size" 
                   | "retention_days" | "metadata" ) , ":" , Value , "\n" ;

MemoryScope     = "user" | "session" | "global" | "buffer" ;
MemoryKind      = "list" | "key_value" | "vector" | "graph" ;
```

### 3.10 Dataset Declaration

```ebnf
DatasetDecl     = "dataset" , QuotedName , DatasetSource , [ DatasetFilter ] , [ DatasetSchema ] ;

DatasetSource   = "from" , ( TableSource | FileSource | QuerySource ) ;
TableSource     = [ DatabaseType ] , "table" , IDENTIFIER ;
FileSource      = "file" , STRING_LITERAL ;
QuerySource     = "query" , STRING_LITERAL ;

DatasetFilter   = ":" , "\n" , INDENT , "filter" , "by" , ":" , Expression , "\n" , DEDENT ;
DatasetSchema   = ":" , "\n" , INDENT , SchemaDefinition , DEDENT ;
```

### 3.11 Function Declaration

```ebnf
FunctionDecl    = "fn" , IDENTIFIER , FunctionSignature , FunctionBody ;

FunctionSignature = "(" , [ ParameterList ] , ")" , [ TypeAnnotation ] ;
ParameterList   = Parameter , { "," , Parameter } ;
Parameter       = IDENTIFIER , [ ":" , TypeAnnotation ] ;

FunctionBody    = "=>" , ( Expression | Block ) ;
```

---

## 4. Expressions

### 4.1 Expression Syntax

```ebnf
Expression      = AssignmentExpr ;

AssignmentExpr  = LogicalOrExpr , [ AssignOp , AssignmentExpr ] ;
AssignOp        = "=" ;

LogicalOrExpr   = LogicalAndExpr , { "||" , LogicalAndExpr } ;
LogicalAndExpr  = EqualityExpr , { "&&" , EqualityExpr } ;
EqualityExpr    = RelationalExpr , { EqualityOp , RelationalExpr } ;
EqualityOp      = "==" | "!=" ;

RelationalExpr  = AdditiveExpr , { RelationalOp , AdditiveExpr } ;
RelationalOp    = "<" | ">" | "<=" | ">=" ;

AdditiveExpr    = MultiplicativeExpr , { AdditiveOp , MultiplicativeExpr } ;
AdditiveOp      = "+" | "-" ;

MultiplicativeExpr = ExponentialExpr , { MultiplicativeOp , ExponentialExpr } ;
MultiplicativeOp = "*" | "/" | "%" ;

ExponentialExpr = UnaryExpr , [ "**" , ExponentialExpr ] ;

UnaryExpr       = [ UnaryOp ] , PostfixExpr ;
UnaryOp         = "!" | "-" | "+" ;

PostfixExpr     = PrimaryExpr , { PostfixOp } ;
PostfixOp       = FunctionCall | MemberAccess | IndexAccess ;

FunctionCall    = "(" , [ ArgumentList ] , ")" ;
ArgumentList    = Expression , { "," , Expression } ;

MemberAccess    = "." , IDENTIFIER ;
IndexAccess     = "[" , Expression , "]" ;
```

### 4.2 Primary Expressions

```ebnf
PrimaryExpr     = Literal
                | IDENTIFIER
                | LambdaExpr
                | MatchExpr
                | LetExpr
                | ArrayLiteral
                | ObjectLiteral
                | ParenExpr ;

Literal         = STRING_LITERAL | NUMBER | BOOLEAN | NULL ;

LambdaExpr      = "fn" , "(" , [ ParameterList ] , ")" , "=>" , Expression ;

MatchExpr       = "match" , Expression , "{" , { MatchCase } , "}" ;
MatchCase       = "case" , Pattern , "=>" , Expression ;
Pattern         = Literal | ArrayPattern | ObjectPattern | IDENTIFIER ;
ArrayPattern    = "[" , [ PatternList ] , [ "..." , IDENTIFIER ] , "]" ;
PatternList     = Pattern , { "," , Pattern } ;

LetExpr         = "let" , IDENTIFIER , "=" , Expression , "in" , Expression ;

ArrayLiteral    = "[" , [ ExpressionList ] , "]" ;
ExpressionList  = Expression , { "," , Expression } ;

ObjectLiteral   = "{" , [ ObjectFieldList ] , "}" ;
ObjectFieldList = ObjectField , { "," , ObjectField } ;
ObjectField     = ( IDENTIFIER | STRING_LITERAL ) , ":" , Expression ;

ParenExpr       = "(" , Expression , ")" ;
```

---

## 5. Control Flow

### 5.1 Conditional Statements

```ebnf
IfStatement     = "if" , Expression , ":" , Block , [ ElseClause ] ;
ElseClause      = "else" , ":" , Block ;
```

### 5.2 Loop Statements

```ebnf
ForStatement    = "for" , IDENTIFIER , "in" , Expression , ":" , Block ;
WhileStatement  = "while" , Expression , ":" , Block ;
```

---

## 6. Values and Types

### 6.1 Value Types

```ebnf
Value           = Literal
                | ArrayValue
                | ObjectValue
                | FunctionRef
                | EnvRef ;

ArrayValue      = "[" , [ ValueList ] , "]" ;
ValueList       = Value , { "," , Value } ;

ObjectValue     = "{" , [ ObjectPairList ] , "}" ;
ObjectPairList  = ObjectPair , { "," , ObjectPair } ;
ObjectPair      = ( IDENTIFIER | STRING_LITERAL ) , ":" , Value ;

FunctionRef     = IDENTIFIER ;
EnvRef          = "env" , "." , IDENTIFIER ;
```

### 6.2 Type System

```ebnf
Type            = PrimitiveType | ArrayType | ObjectType | FunctionType | UnionType ;

PrimitiveType   = "text" | "number" | "boolean" | "null" | "any" ;
ArrayType       = Type , "[" , "]" ;
ObjectType      = "{" , [ TypeFieldList ] , "}" ;
TypeFieldList   = TypeField , { "," , TypeField } ;
TypeField       = IDENTIFIER , ":" , Type ;

FunctionType    = "(" , [ TypeList ] , ")" , "=>" , Type ;
TypeList        = Type , { "," , Type } ;

UnionType       = Type , { "|" , Type } ;
```

---

## 7. Indentation Rules

N3 uses **significant whitespace** similar to Python:

```ebnf
INDENT          = (* Increase indentation level by 2 or 4 spaces *) ;
DEDENT          = (* Decrease indentation level *) ;

(* Rules:
   1. Blocks must be consistently indented (2 or 4 spaces)
   2. All statements at same nesting level must have same indentation
   3. Top-level declarations must have zero indentation
   4. Comments at any indentation level are ignored
*)
```

---

## 8. Operator Precedence

From highest to lowest:

| Precedence | Operators              | Associativity |
|------------|------------------------|---------------|
| 1          | `()` `[]` `.`          | Left-to-right |
| 2          | `**`                   | Right-to-left |
| 3          | `!` `-` (unary)        | Right-to-left |
| 4          | `*` `/` `%`            | Left-to-right |
| 5          | `+` `-`                | Left-to-right |
| 6          | `<` `>` `<=` `>=`      | Left-to-right |
| 7          | `==` `!=`              | Left-to-right |
| 8          | `&&`                   | Left-to-right |
| 9          | `||`                   | Left-to-right |
| 10         | `=>`                   | Right-to-left |
| 11         | `=`                    | Right-to-left |

---

## 9. Grammar Conformance Notes

### 9.1 Canonical Syntax Requirements

1. **All top-level constructs** MUST use the format: `keyword "name" { ... }`
2. **Block syntax** is mandatory for all declarations (no colon-only syntax)
3. **Quoted names** are required for all user-defined identifiers in declarations
4. **Consistent indentation** (2 or 4 spaces, chosen per-file)
5. **Explicit typing** for function parameters and schemas

### 9.2 Deprecated Syntax (Not Allowed)

The following legacy syntaxes are **NO LONGER SUPPORTED**:

```n3
# ❌ DEPRECATED: Colon-based syntax
llm gpt4:
    provider: openai
    model: gpt-4

# ✅ CANONICAL: Brace-based syntax
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
}

# ❌ DEPRECATED: Unquoted names
agent my_agent { }

# ✅ CANONICAL: Quoted names
agent "my_agent" { }
```

---

## 10. Examples

### 10.1 Complete Application

```n3
module "my_app"

import ai.models as models

language_version: "1.0"

app "Customer Service Bot" connects to postgres "customer_db" {
  description: "AI-powered customer support"
  version: "1.0.0"
}

llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 1000
}

memory "conversation_history" {
  scope: "session"
  kind: "list"
  max_items: 50
}

agent "support_agent" {
  llm: "gpt4"
  system_prompt: "You are a helpful customer service agent."
  tools: ["search_kb", "create_ticket"]
  memory: "conversation_history"
}

page "Chat" at "/chat" {
  show text {
    title: "Customer Support"
  }
  show chat {
    agent: "support_agent"
  }
}
```

### 10.2 RAG Pipeline

```n3
index "docs_index" {
  source_dataset: "documentation"
  embedding_model: "text-embedding-3-small"
  chunk_size: 512
  overlap: 64
  backend: "pgvector"
}

rag_pipeline "doc_retrieval" {
  query_encoder: "text-embedding-3-small"
  index: "docs_index"
  top_k: 5
  distance_metric: "cosine"
}

chain "rag_qa_chain" {
  input -> rag "doc_retrieval" -> prompt "doc_qa" -> llm "gpt4"
}
```

---

## Appendix A: Railroad Diagrams

Railroad diagrams for key grammar rules are available at:
`docs/grammar_diagrams/`

---

## Appendix B: Version History

| Version | Date       | Changes                          |
|---------|------------|----------------------------------|
| 1.0.0   | 2025-11-21 | Initial canonical grammar        |

---

**End of Grammar Specification**
