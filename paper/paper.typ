// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = [
  #line(start: (25%,0%), end: (75%,0%))
]

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.amount
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == "string" {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == "content" {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != "string" {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#"❕"" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: white, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "linux libertine",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "linux libertine",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)
  if title != none {
    align(center)[#block(inset: 2em)[
      #set par(leading: heading-line-height)
      #if (heading-family != none or heading-weight != "bold" or heading-style != "normal"
           or heading-color != black or heading-decoration == "underline"
           or heading-background-color != none) {
        set text(font: heading-family, weight: heading-weight, style: heading-style, fill: heading-color)
        text(size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(size: subtitle-size)[#subtitle]
        }
      } else {
        text(weight: "bold", size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(weight: "bold", size: subtitle-size)[#subtitle]
        }
      }
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)
#import "@preview/ctheorems:1.1.0": *
#show: thmrules
#let definition = thmbox("definition", "Definition", base_level: 1)
#import "@preview/fontawesome:0.1.0": *
#let theorem = thmbox("theorem", "Theorem", base_level: 1)
#let corollary = thmbox("corollary", "Corollary", base_level: 1)

#show: doc => article(
  title: [BlockTensorDecompositions.jl: A Unified Constrained Tensor Decomposition Julia Package],
  authors: (
    ( name: [Nicholas J. E. Richardson],
      affiliation: [Department of Mathematics],
      email: [] ),
    ( name: [Noah Marusenko],
      affiliation: [Department of Computer Science],
      email: [] ),
    ( name: [Michael P. Friedlander],
      affiliation: [Departments of Mathematics and Computer Science

],
      email: [] ),
    ),
  font: ("Libertinus Serif",),
  sectionnumbering: "1.1.a.i",
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)

= Introduction
<introduction>
- Tenors are useful in many applications
- Need tools for fast and efficient decompositions

For the scientific user, it would be most useful for there to be a single piece of software that can take as input 1) any reasonable type of factorization model and 2) constraints on the individual factors, and produce a factorization. Details like what rank to select, how the constraints should be enforced, and convergence criteria should be handled automatically, but customizable to the knowledgable user. These are the core specification for BlockTensorDecompositions.jl.

== Related tools
<related-tools>
- Packages within Julia
- Other languages
- Hint at why I developed this

Beyond the external usefulness already mentioned, this package offers a playground for fair comparisons of different parameters and options for performing tensor factorizations across various decomposition models. There exist packages for working with tensors in languages like Python (TensorFlow @martin_abadi_tensorflow_2015, PyTorch @ansel_pytorch_2024, and TensorLy @kossaifi_tensorly_2019), MATLAB (Tensor Toolbox @bader_tensor_2023), R (rTensor @li_rtensor_2018), and Julia (TensorKit.jl @jutho_juthotensorkitjl_2024, Tullio.jl @abbott_mcabbotttulliojl_2023, OMEinsum.jl @peter_under-peteromeinsumjl_2024, and TensorDecompositions.jl @wu_yunjhongwutensordecompositionsjl_2024). But they only provide a groundwork for basic manipulation of tensors and the most common tensor decomposition models and algorithms, and are not equipped to handle arbitrary user defined constraints and factorization models.

Some progress towards building a unified framework has been made @xu_BlockCoordinateDescent_2013@kim_algorithms_2014@yang_unified_2011. But these approaches don’t operate on the high dimensional tensor data natively and rely on matricizations of the problem, or only consider nonnegative constraints. They also don’t provide an all-in-one package for executing their frameworks.

== Contributions
<contributions>
- Fast and flexible tensor decomposition package
- Framework for creating and performing custom
  - tensor decompositions
  - constrained factorization (the what)
  - iterative updates (the how)
- Implement new "tricks"
  - a (Lipschitz) matrix step size for efficient sub-block updates
  - multi-scaled factorization when tensor entries are discretizations of a continuous function
  - partial projection and rescaling to enforce linear constraints (rather than Euclidean projection)
- ?? rank detection ??

The main contribution is a description of a fast and flexible tensor decomposition package, along with a public implementation written in Julia: BlockTensorDecompositions.jl. This package provides a framework for creating and performing custom tensor decompositions. To the author’s knowledge, it is the first package to provide automatic factorization to a large class of constrained tensor decompositions problems, as well as a framework for implementing new constraints and iterative algorithms. This paper also describes three new techniques not found in the literature that empirically convergence faster than traditional block-coordinate descent.

= Tensor Decompositions
<tensor-decompositions>
- the math section of the paper

This section reviews the notation used throughout the paper and commonly used tensor decompositions.

== Notation
<notation>
- tensor notation, use MATLAB notation for indexing so subscripts can be used for a sequence of tensors

=== Sets
<sets>
The set of real number is denoted as $bb(R)$ and its restrictions to nonnegative numbers is denoted as $bb(R)_(+) = bb(R)_(gt.eq 0) = {x in bb(R) mid(bar.v) x gt.eq 0}$.

We use $[N] = { 1 , 2 , dots.h , N } = { n }_(n = 1)^N$ to denote integers from $1$ to $N$.

Usually, lower case symbols will be used for the running index, and the capitalized letter will be the maximum letter it runs to. This leads to the convenient shorthand $i in [I]$, $j in [J]$, etc.

We use a capital delta $Delta$ to denote sets of vectors or higher order tensors where the slices or fibres along a specified dimension sum to $1$, i.e.~generalized simplexes.

Usually, we use script letters ($cal(A) , cal(B) , cal(C) ,$ etc.) for other sets.

=== Vectors, Matrices, and Tensors
<vectors-matrices-and-tensors>
Vectors are denoted with lowercase letters ($x$, $y$, etc.), and matrices and higher order tensors with uppercase letters (commonly $A$, $B$, $C$ and $X$, $Y$, $Z$). The order of a tensor is the number of axes it has. We would call vectors "order-1" or "1st order" tensors, and matrices "order-2" or "2nd order" tensors.

To avoid confusion between entries of a vector/matrix/tensor and indexing a list of objects, we use square brackets to denote the former, and subscripts to denote the later. For example, the entry in the $i$th row and $j$th column of a matrix $A in bb(R)$ is $A [i , j]$. This follows MATLAB/Julia notation where `A[i,j]` points to the entry $A [i , j]$. We contrast this with a list of $I$ objects being denoted as $a_1 , dots.h , a_I$, or more compactly, ${ a_i }$ when it is clear the index $i in [I]$.

The transpose $A^tack.b in bb(R)^(J times I)$ of a matrix $A in bb(R)^(I times J)$ flips entries along the main diagonal: $A^tack.b [j , i] = A [i , j]$. In Julia, the transpose of a matrix is typed with a single apostrophe `A'`.

The $n$-slices, $n$th mode slices, or mode $n$ slices of an $N$th order tensor $A$ are notated with the slice $A [: , med dots.h , med : , med i_n , med : , med dots.h , med :]$. For a $3$rd order tensor $A$, the $1$st, $2$nd, and $3$rd mode slices $A [i , : , :]$, $A [: , j , :]$, and $A [: , : , k]$ have special names and are called the horizontal, lateral, and frontal slices and are displayed in @fig-tensor-slices. In Julia, the 1-, 2-, and 3-slices of a third order array `A` would be `eachslice(A, dims=1)`, `eachslice(A, dims=2)`, and `eachslice(A, dims=3)`.

#figure([
#box(image("figure/tensor_slices.png"))
], caption: figure.caption(
position: bottom, 
[
Slices of an order $3$ tensor $A$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tensor-slices>


The $n$-fibres, $n$th mode fibres, or mode $n$ fibres of an $N$th order tensor $A$ are denoted $A [i_1 , med dots.h , med i_(n - 1) , med : , med i_(n + 1) , med dots.h , med i_N]$. For example, the 1-fibres of a matrix $M$ are the column vectors \
$M [: , med j]$, and the 2-fibres are the row vectors $M [i , med :]$. For order-3 tensors, the $1$st, $2$nd, and $3$rd mode fibres $A [: , j , k]$, $A [i , : , :]$, and $A [i , j , :]$ are called the vertical/column, horizontal/row, and depth/tube fibres respectively and are displayed in @fig-tensor-fibres. Natively in Julia, the 1-, 2-, and 3-fibres of a third order array `A` would be `eachslice(A, dims=(2,3))`, `eachslice(A, dims=(1,3))`, and `eachslice(A, dims=(1,2))`. BlockTensorDecomposition.jl defines the function `eachfibre(A; n)` to do exactly this. For example, the 1-fibres of an array `A` would be `eachfibre(A, n=1)`.

For matrices, the 1-fibres are the same as the 2-slices (and vice versa), but for $N$th order tensors in general, fibres are always vectors, whereas $n$-slices are $(N - 1)$th order tensors.

#figure([
#box(image("figure/tensor_fibres.png"))
], caption: figure.caption(
position: bottom, 
[
Fibres of an order $3$ tensor $A$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tensor-fibres>


Since we commonly use $I$ as the size of a tensor’s dimension, we use $upright(i d)_I$ to denote the identity tensor of size $I$ (of the appropriate order). When the order is $2$, $upright(i d)_I$ is an $I times I$ matrix with ones along the main diagonal, and zeros elsewhere. For higher orders $N$, this is an $underbrace(I times dots.h.c times I, N upright("times"))$ tensor where $upright(i d)_I [i_1 , dots.h , i_N] = 1$ when $i_1 = dots.h = i_N in [I]$, and is zero otherwise.

BlockTensorDecomposition.jl defines `identity_tensor(I, ndims)` to construct $upright(i d)_I$.

=== Products of Tensors
<products-of-tensors>
#definition()[
The outer product $times.circle$ between two tensors $A in bb(R)^(I_1 times dots.h.c times I_M)$ and $B in bb(R)^(J_1 times dots.h.c times J_N)$ yields an order $M + N$ tensor $A times.circle B in bb(R)^(I_1 times dots.h.c times I_M times J_1 times dots.h.c times J_N)$ that is entry-wise

$ (A times.circle B) [i_1 , dots.h , i_M , j_1 , dots.h , j_N] = A [i_1 , dots.h , i_M] B [j_1 , dots.h , j_N] . $

] <def-outer-product>
TODO Define in BlockTensorDecomposition.jl

The Frobenius inner product between two tensors $A , B in bb(R)^(I_1 times dots.h.c times I_N)$ yields a real number $A dot.op B in bb(R)$ and is defined as

$ ⟨A , B⟩ = A dot.op B = sum_(i_1 = 1)^(I_1) dots.h sum_(i_N = 1)^(I_N) A [i_1 , dots.h , i_N] B [i_1 , dots.h , i_N] . $

Julia’s standard library package LinearAlgebra implements the Frobenius inner product with `dot(A, B)` or `A ⋅ B`.

The $n$-slice dot product $dot.op_n$ between two tensors $A in bb(R)^(K_1 , dots.h , K_(n - 1) , I , K_(n + 1) , dots.h , K_N)$ and $B in bb(R)^(K_1 , dots.h , K_(n - 1) , J , K_(n + 1) , dots.h , K_N)$ returns a matrix $(A dot.op_n B) in bb(R)^(I times J)$ with entries

$ (A dot.op_n B) [i , j] = sum_(k_1 dots.h k_(n - 1) k_(n + 1) dots.h k_N) A [k_1 , dots.h , k_(n - 1) , i , k_(n + 1) , dots.h , k_N] B [k_1 , dots.h , k_(n - 1) , j , k_(n + 1) , dots.h , k_N] . $

This product can also be thought of as taking the dot product $(A dot.op_n B) [i , j] = A_i dot.op B_j$ between all pairs of $n$th order slices of $A$ and $B$, which exactly how BlockTensorDecomposition.jl defines the operation.

```julia
function slicewise_dot(A::AbstractArray, B::AbstractArray; dims=1)
    C = zeros(size(A, dims), size(B, dims))
    if A === B # use faster routine if they are the same
        return _slicewise_self_dot!(C, A; dims)
    end

    for (i, A_slice) in enumerate(eachslice(A; dims))
        for (j, B_slice) in enumerate(eachslice(B; dims))
            C[i, j] = A_slice ⋅ B_slice
        end
    end
    return C
end

function _slicewise_self_dot!(C, A; dims=1)
    enumerated_A_slices = enumerate(eachslice(A; dims))
    for (i, Ai_slice) in enumerated_A_slices
        for (j, Aj_slice) in enumerated_A_slices
            if i > j
                continue
            else # only compute the upper triangle entries of C
                C[i, j] = Ai_slice ⋅ Aj_slice
            end
        end
    end
    return Symmetric(C) # indexing C[2,1] points to the entry in C[1,2]
end
```

BlockTensorDecomposition.jl defines this operation with `slicewise_dot(A, B, n)`. In the special case where $A = B$, a more efficient method that only computes entries where $i lt.eq j$ is defined since $A dot.op_n A$ is a symmetric matrix.

The $n$-slice product of a tensor with itself $X dot.op_n X$ should be thought of as a generalization of the Gram matrix $X^tack.b X$ since it considers the matrix generated by taking the dot product between every $n$th mode slice, just like how the Gram matrix considers the dot product between every pair of columns.

The $n$-mode product $times_n$ between a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ and matrix $B in bb(R)^(I_n times J)$, returns a tensor $(A times_n B) in bb(R)^(I_1 times dots.h.c times I_(n - 1) times J times I_(n + 1) times dots.h.c times I_N)$ with entries

$ (A times_n B) [i_1 , dots.h , i_(n - 1) , j , i_(n + 1) , dots.h , i_N] = sum_(i_n = 1)^(I_n) A [i_1 , dots.h , i_(n - 1) , i_n , i_(n + 1) , dots.h , i_N] B [i_n , j] . $

BlockTensorDecomposition.jl defines this operation with `nmode_product(A, B, n)`.

```julia
function nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)
    # convert the problem to the mode-1 product
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ B
    return swapdims(Cperm, n) # swap back
end

function ×₁(A::AbstractArray, B::AbstractMatrix)
    # Turn the 1-mode product into matrix-matrix multiplication
    sizeA = size(A)
    Amat = reshape(A, sizeA[1], :)

    # Initialize the output tensor
    C = zeros(size(B, 1), sizeA[2:end]...)
    Cmat = reshape(C, size(B, 1), prod(sizeA[2:end]))

    # Perform matrix-matrix multiplication Cmat = B*Amat
    mul!(Cmat, B, Amat)

    return C # Output entries of Cmat in tensor form
end

function swapdims(A::AbstractArray, a::Integer, b::Integer=1)
    # Construct a permutation where a and b are swapped
    # e.g. [4, 2, 3, 1, 5, 6] when a=4 and b=1
    dims = collect(1:ndims(A))
    dims[a] = b; dims[b] = a
    return permutedims(A, dims)
end
```

#block[
#callout(
body: 
[
If we were only working with a fixed order of tensors, we could have defined `×₁` entry-wise with `Tullio.jl`. The function definition `tullio×₁` below gives an example for order three tensors.

```julia
function tullio×₁(A::AbstractArray{_,3}, B::AbstractMatrix)
  @tullio C[i, j, k] := A[r, j, k] * B[i, r]
  return C
end
```

But we would need a new definition for each ordered tensor, or use Julia’s meta programming to write a method for each order at runtime.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
)
]
The $n$-mode product and $n$-slice product can be thought of as opposites of each other. The $n$-mode product sums over just the $n$th dimension of the first tensor, whereas the $n$-slice product sums over all but the $n$th dimension.

We can extend the $n$-mode product to sum over multiple indices between two tensors.

The multi-mode product $times_(1 , dots.h , n) = times_(1 : n) = times_([n])$ between a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ and tensor $B in bb(R)^(I_1 times dots.h.c times I_n)$, returns a tensor $(A times_([n]) B) in bb(R)^(I_(n + 1) times dots.h.c times I_N)$ with entries

$ (A times_([n]) B) [i_(n + 1) , dots.h , i_N] = sum_(i_1 , dots.h , i_n) A [i_1 , dots.h , i_n , i_(n + 1) , dots.h , i_N] B [i_1 , dots.h , i_n] . $

This product contracts the first $n$ indexes of $A$ with every index of $B$.

More generally, we can contract any number of indexes such as the last $n$ indexes of $A$ with every index of $B$ with $times_(N - n + 1 , dots.h , N) = times_((N - n + 1) : N) = times_([- n])$,

$ (A times_([- n]) B) [i_1 , dots.h , i_n] = sum_(i_(n + 1) , dots.h , i_N) A [i_1 , dots.h , i_(N - n) , i_(N - n + 1) , dots.h , i_N] B [i_(N - n + 1) , dots.h , i_N] , $

or specific indexes. For example, we would define $(A times_(1 , 3 , 5) B) in bb(R)^(I_2 times I_4 times I_6)$ where $A in bb(R)^(I_1 times dots.h.c times I_6)$ and $B in bb(R)^(I_1 times I_3 times I_5)$ to be

$ (A times_(1 , 3 , 5) B) [i_2 , i_4 , i_6] = sum_(i_1 , i_3 , i_5) A [i_1 , i_2 , i_3 , i_4 , i_5 , i_6] B [i_1 , i_3 , i_5] . $

When $A$ a #emph[half-symmetric];#footnote[For example, the Hessian of a scalar function (see @def-hessian).] tensor of order $2 N$

#math.equation(block: true, numbering: "(1)", [ $ A [i_1 , dots.h , i_N , i_(N + 1) , dots.h , i_(2 N)] = A [i_(N + 1) , dots.h , i_(2 N) , i_1 , dots.h , i_N] , $ ])<eq-half-symmetric>

we have

$ A times_([N]) B = A times_([- N]) B $

for tensors $B$ of order $N$.

=== Gradients, Norms, and Lipschitz Constants
<gradients-norms-and-lipschitz-constants>
#definition("Gradient")[
The gradient $nabla f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^(I_1 times dots.h.c times I_N)$ of a (differentiable) function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is defined entry-wise for a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ by

$ nabla f (A) [i_1 , dots.h , i_N] = frac(partial f, partial A [i_1 , dots.h , i_N]) (A) . $

] <def-gradient>
#definition("Hessian")[
The Hessian $nabla^2 f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^((I_1 times dots.h.c times I_N)^2)$ of a second-differentiable function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is the gradient of the gradient and is defined for a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ entry-wise by

$ nabla^2 f (A) [i_1 , dots.h , i_N , j_1 , dots.h , j_N] = frac(partial^2 f, partial A [i_1 , dots.h , i_N] partial A [j_1 , dots.h , j_N]) (A) . $

] <def-hessian>
For a tensor input $A$ of order $N$, the Hessian tensor $nabla^2 f (A)$ is of order $2 N$.

See @sec-hessian-from-gradient for how this definition can be reproduced by performing two gradients $nabla^2 f = nabla (nabla f)$.

#definition()[
The Frobenius norm of a tensor $A$ is the square root of its dot product with itself

$ norm(A)_F = sqrt(⟨A , A⟩) . $

] <def-frobenius-norm>
For vectors $v$, this is equivalent to the (Euclidean) 2-norm

$ norm(v)_F = norm(v)_2 = sqrt(⟨v , v⟩) . $

For matrices $M$, the ($2 arrow.r 2$) operator norm is defined as

$ norm(M)_(upright("op")) = sup_(lr(bar.v.double v bar.v.double)_2 = 1) norm(M v)_2 = sigma_1 (M) $

where $sigma_1 (M)$ is the largest singular value of $M$.

For tensors $T$, the operator-norm is ambiguous since there are multiple ways we can treat tensors as function on other tensors. There is a canonical way to do this for vectors $x arrow.r.bar v^tack.b x$ and matrices $x arrow.r.bar M x$, but not tensors. In the case of the Hessian tensor $nabla^2 f (A) in bb(R)^((I_1 times dots.h.c times I_N)^2)$ evaluated at $A in bb(R)^(I_1 times dots.h.c times I_N)$, it is natural to consider the function $X arrow.r.bar nabla^2 f (A) times_([N]) X$ for $X in bb(R)^(I_1 times dots.h.c times I_N)$. This gives us our definition of the operator norm on tensors.

#definition("Operator Norm")[
The operator norm of a half-symmetric tensor $A in bb(R)^((I_1 times dots.h.c times I_N)^2)$ (@eq-half-symmetric) is defined as

#math.equation(block: true, numbering: "(1)", [ $ norm(A)_(upright("op")) = sup_(norm(X)_F = 1) norm(A times_([N]) X)_F . $ ])<eq-operator-norm>

In @eq-operator-norm, $X in bb(R)^(I_1 times dots.h.c times I_N)$. Note that this definition agrees with the usual operator norm on matrices when $N = 1$.

] <def-operator-norm>
#theorem("Norm of an outer product")[
Let $T = A_1 times.circle dots.h.c times.circle A_N in bb(R)^((I_1 times dots.h.c times I_N)^2)$ where $A_n in bb(R)^(I_n times I_n)$ are symmetric matrices.

Then $T$ is half-symmetric and

$ norm(T)_(upright("op")) = product_(n = 1)^N norm(A_n)_(upright("op")) . $

] <thm-operator-norm-outer-product>
#block[
#callout(
body: 
[
According to how the outer product $times.circle$ is defined in @def-outer-product, the product $A_1 times.circle dots.h.c times.circle A_N$ shown in @thm-operator-norm-outer-product is really an element of $bb(R)^(I_1 times I_1 times dots.h.c times I_N times I_N)$. Note how the indexes are ordered differently than an element of $bb(R)^((I_1 times dots.h.c times I_N)^2) = bb(R)^(I_1 times dots.h.c times I_N times I_1 times dots.h.c times I_N)$. Correcting for this with explicit notation becomes cumbersome and would require tensor transposes, a new definition of an outer product, or reordering of indexes in the definition of a half-symmetric tensor. These can have knock-on effects to the definition of the Hessian, multi-mode product, and the operator norm.

To avoid the headache, the equality

$ T = A_1 times.circle dots.h.c times.circle A_N $

in @thm-operator-norm-outer-product should be thought of as the following entry-wise equation

#math.equation(block: true, numbering: "(1)", [ $ T [i_1 , dots.h , i_N , j_1 , dots.h , j_N] = A_1 [i_1 , j_1] dots.h.c A_N [i_N , j_N] . $ ])<eq-corrected-outer-product>

With the outer product understood as @eq-corrected-outer-product, the results of @thm-operator-norm-outer-product that $T$ is half-symmetric and its operator norm is the product of the operator norms of the constituent matrices is true.

]
, 
title: 
[
Warning
]
, 
background_color: 
rgb("#fcefdc")
, 
icon_color: 
rgb("#EB9113")
, 
icon: 
"⚠"
)
]
#definition("Lipschitz Function")[
A function $g : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^(I_1 times dots.h.c times I_N)$ is $L$-Lipschitz when

$ norm(g (A) - g (B))_F lt.eq L norm(A - B)_F , quad forall A , B in bb(R)^(I_1 times dots.h.c times I_N) . $

We call the smallest such $L$ #emph[the] Lipschitz constant of $g$.

] <def-lipschitz>
#definition("Smooth Function")[
A differentiable function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is $L$-smooth when its gradient $g = nabla f$ is $L$-Lipschitz.

] <def-smooth>
#theorem("Quadratic Smoothness")[
Let $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ be a quadratic function

$ f (X) = 1 / 2 A (X , X) + B (X) + C $

of $X$ with bilinear function $A : (bb(R)^(I_1 times dots.h.c times I_N))^2 arrow.r bb(R)$, linear function $B : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$, and constant $C in bb(R)^(I_1 times dots.h.c times I_N)$.

Then:

+ the Hessian $nabla^2 f$ is a constant function that evaluates to $nabla^2 f (X) = D$ at every point $X$ for some $D in bb(R)^((I_1 times dots.h.c times I_N)^2)$,
+ the tensor $D$ only depends on the bilinear function $A$ (and not on $B$ and $C$), and
+ the quadratic function $f$ is $L$-smooth with constant

$ L = norm(D)_(upright("op")) . $

] <thm-quadratic-smoothness>
== Common Decompositions
<sec-common-decompositions>
- Extensions of PCA/ICA/NMF to higher dimensions
- talk about the most popular Tucker, Tucker-n, CP
- other decompositions
  - high order SVD (see Kolda and Bader)
  - HOSVD (see Kolda, Shifted power method for computing tensor eigenpairs)

A tensor decomposition is a factorization of a tensor into multiple (usually smaller) tensors, that can be recombined into the original tensor. To make a common interface for decompositions, we make an abstract subtype of Julia’s `AbstractArray`, and subtype `AbstractDecomposition` for our concrete tensor decompositions.

```julia
abstract type AbstractDecomposition{T, N} <: AbstractArray{T, N} end
```

Computationally, we can think of a generic decomposition as storing factors $(A , B , C , . . .)$ and operations $(times_a , times_b , . . .)$ for combining them. This is what we do in BlockTensorDecomposition.jl.

```julia
struct GenericDecomposition{T, N} <: AbstractDecomposition{T, N}
    factors::Tuple{Vararg{AbstractArray{T}}} # e.g. (A, B, C)
    contractions::Tuple{Vararg{Function}} # e.g. (×₁, ×₂)
end
# Y = A ×₁ B ×₂ C
array(G::GenericDecomposition) = multifoldl(contractions(G), factors(G))
```

The function `multifoldl` applies the given operations between each factor, from left to right.

```julia
function multifoldl(ops, args)
    @assert (length(ops) + 1) == length(args)
    x, xs... = args
    for (op, arg) in zip(ops, xs)
        x = op(x, arg)
    end
    return x
end
```

Different types of decompositions define different operations, and different "ranks" of the same decomposition specific the sizes of the factors used.

A commonly used family of decompositions can be derived from the Tucker decomposition.

#definition()[
A rank-$(R_1 , dots.h , R_N)$ Tucker decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $N$ matrices $A_n in bb(R)^(I_n times R_n)$, $n in [N]$, and core tensor $B in bb(R)^(R_1 times dots.h.c times R_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 , dots.h , i_N] = sum_(r_1 = 1)^(R_1) dots.h sum_(r_N = 1)^(R_N) A_1 [i_1 , r_1] dots.h.c A_r [i_N , r_N] B [r_1 , dots.h , r_N] $ ])<eq-tucker>

entry-wise. More compactly, this decomposition can be written using the $n$-mode product, or with double brackets

#math.equation(block: true, numbering: "(1)", [ $ Y = B times_1 A_1 times_2 dots.h times_N A_N = B times.big_n A_n = lr(bracket.l.double B \; A_1 , dots.h , A_N bracket.r.double) . $ ])<eq-tucker-product>

] <def-tucker-decomposition>
The #emph[Tucker Product] defined by @eq-tucker-product is implemented in BlockTensorDecomposition.jl with `tuckerproduct(B, (A1, ..., AN))` and computes $ B times.big_n A_n = lr(bracket.l.double B \; A_1 , dots.h , A_N bracket.r.double) . $

It can also optionally "exclude" one of the matrix factors with the call `tuckerproduct(B, (A1, ..., AN); exclude=n)` to compute

$ B times.big_(m eq.not n) A_m = lr(bracket.l.double B \; A_1 , dots.h , A_(n - 1) , upright("id")_(R_n) , A_(n + 1) , dots.h , A_N bracket.r.double) . $

```julia
function tuckerproduct(core, matrices; exclude=nothing)
    N = ndims(core)
    if isnothing(exclude)
        return multifoldl(tucker_contractions(N), (core, matrices...))
    else
        return multifoldl(getnotindex(tucker_contractions(N), exclude), (core, getnotindex(matrices, exclude)...))
    end
end

tucker_contractions(N) = Tuple((B, A) -> nmode_product(B, A, n) for n in 1:N)
```

Sometimes we write $A_0 = B$ to ease notation, and suggest the "zeroth" factor of the tucker decomposition is the core tensor $B$. In the special case when $N = 3$, we can visualize Tucker decomposition as multiplying the core tensor by matrices on all three sides as shown in @fig-tucker.

#figure([
#box(image("figure/tucker_decomposition_order_3.png"))
], caption: figure.caption(
position: bottom, 
[
Tucker factorization of a $3$rd order tensor $Y$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tucker>


Setting all the matrices of a Tucker decomposition to the identity matrix but the first gives the Tucker-$1$ decomposition.

#definition()[
A rank-$R$ Tucker-$1$ decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces a matrix $A in bb(R)^(I_1 times R)$, and core tensor $B in bb(R)^(R times I_2 times dots.h.c times I_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 , dots.h , i_N] = sum_(r = 1)^R A [i_1 , r] B [r , i_2 , dots.h , i_N] $ ])<eq-tucker-1>

entry-wise or more compactly,

$ Y = A B = B times_1 A = lr(bracket.l.double B \; A bracket.r.double) . $

] <def-tucker-1-decomposition>
Note we extend the usual definition of matrix-matrix multiplication

$ (A B) [i , j] = sum_(r = 1)^R A [i , r] B [r , j] $

to tensors $B$ in the compact notation for Tucker-1 decomposition $Y = A B$.

More generally, any number of matrices can be set to the identity matrix giving the Tucker-$n$ decomposition.

#definition()[
A rank-$(R_1 , dots.h , R_n)$ Tucker-$n$ decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $n$ matrices $A_1 , dots.h , A_n$, and core tensor $B in bb(R)^(R_1 times dots.h.c times R_n times I_(n + 1) times dots.h.c times I_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 , dots.h , i_N] = sum_(r_1 = 1)^(R_1) dots.h sum_(r_N = 1)^(R_n) A_1 [i_1 , r_1] dots.h.c A_n [i_N , r_n] B [r_1 , dots.h , r_n , i_(n + 1) , dots.h , i_N] $ ])<eq-tucker-n>

entry-wise, or compactly written in the following three ways, $ Y & = B times_1 A_1 times_2 dots.h times_n A_n times_(n + 1) upright(i d)_(I_(n + 1)) times_(n + 2) dots.h times_N upright(i d)_(I_N)\
Y & = B times_1 A_1 times_2 dots.h times_n A_n\
Y & = lr(bracket.l.double B \; A_1 , dots.h , A_n bracket.r.double) . $

] <def-tucker-n-decomposition>
Lastly, if we set the core tensor $B$ to the identity tensor $upright(i d)_R$, we obtain the #strong[can];onical #strong[decomp];osition/#strong[para];llel #strong[fac];tors model (CANDECOMP/PARAFAC or CP for short).

#definition()[
A rank-$R$ CP decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $N$ matrices $A_n in bb(R)^(I_n times R)$, such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 , dots.h , i_N] = sum_(r = 1)^R A_1 [i_1 , r] dots.h.c A_r [i_N , r] $ ])<eq-cp>

entry-wise. More compactly, this decomposition can be written using the $n$-mode product, or with double brackets

$ Y = upright(i d)_R times_1 A_1 times_2 dots.h times_N A_N = upright(i d)_R times.big_n A_n = lr(bracket.l.double A_1 , dots.h , A_N bracket.r.double) . $

] <def-cp-decomposition>
Note CP decomposition is sometimes referred to as Kruskal decomposition, and requires the core only be diagonal (and not necessarily identity) and the factors $A_n$ have normalized columns $norm(A_n [: , r])_2 = 1$.

Other factorization models are used that combine aspects of CP and Tucker decomposition @kolda_TensorDecompositionsApplications_2009, are specialized for order $3$ tensors @qi_TripleDecompositionTensor_2020@wu_manifold_2022, or provide alternate decomposition models entirely like tensor-trains @oseledets_tensor-train_2011. But the (full) Tucker, and its special cases Tucker-$n$, and CP decomposition are most commonly used extensions of the low-rank matrix factorization to tensors. These factorizations are summarized in @tbl-tensor-factorizations.

#figure([
#table(
  columns: (15%, 25%, 35%, 25%),
  align: (auto,auto,auto,auto,),
  table.header([Name], [Bracket Notation], [$n$-mode Product], [Entry-wise],),
  table.hline(),
  [Tucker], [$lr(bracket.l.double A_0 \; A_1 , dots.h , A_N bracket.r.double)$], [$A_0 times_1 A_1 times_2 dots.h times_N A_N$], [@eq-tucker],
  [Tucker-$1$], [$lr(bracket.l.double A_0 \; A_1 bracket.r.double)$], [$A_0 times_1 A_1$], [@eq-tucker-1],
  [Tucker-$n$], [$lr(bracket.l.double A_0 \; A_1 , dots.h , A_n bracket.r.double)$], [$A_0 times_1 A_1 times_2 dots.h times_n A_n$], [@eq-tucker-n],
  [CP], [$lr(bracket.l.double A_1 , dots.h , A_N bracket.r.double)$], [$upright(i d)_R times_1 A_1 times_2 dots.h times_N A_N$], [@eq-cp],
)
], caption: figure.caption(
position: top, 
[
Summary of common tensor factorizations. Here, $N$ is the order of the factorized tensor.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-tensor-factorizations>


TODO add discussion on other decompositions - high order SVD (see Kolda and Bader) - HOSVD (see Kolda, Shifted power method for computing tensor eigenpairs)

Tensor decompositions are not nessisarily unique. It should be clear that scaling one factor by $x eq.not 0$ and dividing another by $x$ yields the same original tensor. Furthermore, fibres and slices can be permuted without affecting the the original tensor. Up to these manipulations, for a fixed rank, there exist criteria that ensures their decompositions are unique @kolda_TensorDecompositionsApplications_2009@kruskal_three-way_1977@bhaskara_uniqueness_2014.

=== Representing Tucker Decompositions
<representing-tucker-decompositions>
There are implemented in BlockTensorDecomposition.jl and can be called, for a third order tensor, with `Tucker((B, A₁, A₂, A₃))`, `Tucker1((B, A₁))`, and `CPDecomposition((A₁, A₂, A₃))`. These Julia `structs` store the tensor in its factored form. We could define the contractions for these types and use the common interface provided by `array`, but it turns out we can reconstruct the whole tensor more efficiently. If the recombined tensor or particular entries are requested, Julia dispatches on the type of decomposition and calls a particular method of `array` or `getindex`. The implementations for efficient array construction and index access are provided below.

```julia
array(T::Tucker) = multifoldl(tucker_contractions(ndims(T)), factors(T))
tucker_contractions(N) = Tuple((G, A) -> nmode_product(G, A, n) for n in 1:N)
```

TODO add getindex method for Tucker type

```julia
function array(T::Tucker1)
    B, A = factors(T)
    return B ×₁ A
end

function getindex(T::Tucker1, I::Vararg{Int})
    B, A = factors(T)
    i, J... = I # (i, J) = (I[1], I[begin+1:end])
    return (@view A[i, :]) ⋅ view(B, :, J...)
end
```

```julia
array(CPD::CPDecomposition) =
  mapreduce(vector_outer, +, zip((eachcol.(factors(CPD)))...))

vector_outer(v) = reshape(kron(reverse(v)...),length.(v))

getindex(CPD::CPDecomposition, I::Vararg{Int}) =
  sum(reduce(.*, (@view f[i,:]) for (f,i) in zip(factors(CPD), I)))
```

== Tensor rank
<tensor-rank>
- tensor rank
- constrained rank (nonnegative etc.)

The rank of a matrix $Y in bb(R)^(I times J)$ can be defined as the smallest $R in bb(Z)_(+)$ such that there exists an exact factorization $Y = A B$ for some $A in bb(R)^(I times R)$ and $B in bb(R)^(R times J)$.

Although this can be extended to higher order tensors, we must specify under which factorization model we are using. For example, the #emph[CP-rank] $R$ of a tensor $Y$ is the smallest such $R$ that omits an exact CP decomposition of $Y$.

#definition()[
The CP rank of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ is the smallest $R$ such that there exist factors $A_n in bb(R)^(I_n times R)$ and $Y = lr(bracket.l.double A_1 , dots.h , A_N bracket.r.double)$, $ upright("rank")_(upright("CP")) (Y) = min {R mid(bar.v) exists A_n in bb(R)^(I_n times R) , thin n in [N] quad upright("s.t.") quad Y = lr(bracket.l.double A_1 , dots.h , A_N bracket.r.double)} . $

] <def-cp-rank>
In a similar way, we can define the #emph[Tucker-1-rank] $R$.

#definition()[
The Tucker-1 rank of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ is the smallest $R$ such that there exist factors $A in bb(R)^(I_1 times R)$ and $B in bb(R)^(R times I_2 times dots.h.c times I_N)$ where $Y = A B$

$ upright("rank")_(upright("Tucker-1")) (Y) = min {R mid(bar.v) exists A_n in bb(R)^(I_n times R) , B in bb(R)^(R times I_2 times dots.h.c times I_N) thin quad upright("s.t.") quad Y = A B} $

] <def-tucker-1-rank>
For the Tucker and Tucker-$n$ decompositions, we instead call a particular factorization #strong[a] rank-$(R_1 , dots.h , R_N)$ Tucker factorization or #strong[a] rank-$(R_1 , dots.h , R_n)$ Tucker-$n$ factorization, rather than #strong[the] CP- or Tucker-$1$-rank of a tensor or #strong[the] rank of a matrix.

One reason CP and Tucker-$1$ only need a single rank $R$ can be explained by considering the case when the order of the tensor $N = 2$ (matrices). The two factorizations become equivalent and are equal to low-rank $R$ matrix factorization $Y = A B$. In fact, Tucker-$1$ is always equivalent to a low-rank matrix factorization, if you consider a flattening of the tensor to arrange the entries as a matrix.

The idea of tensor rank can be generalized further to constrained rank. These are the smallest rank $R$ such that the factors in the decomposition obey the given set of constraints.

For example, the nonnegative Tucker-1 rank is defined as $ upright("rank")_(upright("Tucker-1"))^(+) (Y) = min {R mid(bar.v) exists A_n in bb(R)_(+)^(I_n times R) , B in bb(R)_(+)^(R times I_2 times dots.h.c times I_N) thin quad upright("s.t.") quad Y = A B} . $

More restrictive constraints increase the rank of the tensor since there is less freedom in selecting the factors.

Most tensor decomposition algorithms require the rank as input \[CITE\] since calculating the rank of the tensor can be NP-hard in general @vavasis_complexity_2010. For applications where the rank is not known a priori, a common strategy is to attempt a decomposition for a variety of ranks, and select the model with smallest rank that still achieves good fit between the factorization and the original tensor.

= Computing Decompositions
<computing-decompositions>
- Given a data tensor and a model, how do we fit the model?

Many tensor decompositions algorithms exist in the literature. Usually, they cyclically (or in a random order) update factors until their reconstruction satisfies some convergence criterion. The base algorithm described in @sec-base-algorithm provides flexible framework for wide class of constrained tensor factorization problems. This framework was selected based on empirical observations where it outperforms other similar algorithms, and has also been observed in the literature @xu_BlockCoordinateDescent_2013.

== Optimization Problem
<optimization-problem>
- Least squares (can use KL, 1 norm, etc.)

Ideally, we would be given a data tensor $Y$ and decomposition model, and compute an exact factorization of $Y$ into its factors. Because there is often measurement, numerical, or modeling error, an exact factorization of $Y$ for a particular rank may not exist. To over come this, we instead try to fit the model to the data. Let $X$ be the reconstruction of factors $A_1 , dots.h , A_N$ according to some decomposition for a fixed rank. We assume we know the size of the factors $A_1 , dots.h , A_N$ and how they are combined to produce a tensor the same size of $Y$, i.e.~the map $g : (A_1 , dots.h , A_N) arrow.r.bar X$.

There are many loss functions that can be used to determine how close the model $X$ is to the data $Y$. In principle, any distance or divergence $d (Y , X)$ could be used. We use the $L_2$ loss or least-squares distance between the tensors $norm(X - Y)_F^2$, but other losses are used for tensor decomposition in practice such as the KL divergence \[CITE\].

The main optimization we must solve is now given.

#definition()[
The constrained least-squares tensor factorization problem is to solve

#math.equation(block: true, numbering: "(1)", [ $ min_(A_1 , dots.h , A_N) 1 / 2 norm(g (A_1 , dots.h , A_N) - Y)_F^2 quad upright("s.t.") quad (A_1 , dots.h , A_N) in cal(C)_1 times dots.h.c times cal(C)_N $ ])<eq-constrained-least-squares>

for a given data tensor $Y$, constraints $cal(C)_1 , dots.h , cal(C)_N$, and decomposition model $g$ with fixed rank.

] <def-constrained-least-squares>
Note the problem would have the same solutions as simply using the objective $norm(g (A_1 , dots.h , A_N) - Y)$ without squaring and dividing by $2$. We define the objective in @eq-constrained-least-squares to make computing the function value and gradients faster.

== Base algorithm
<sec-base-algorithm>
- Use Block Coordinate Descent / Alternating Proximal Descent
  - do #emph[not] use alternating least squares (slower for unconstrained problems, no closed form update for general constrained problems)

Let $f (A_1 , dots.h , A_N) := 1 / 2 norm(g (A_1 , dots.h , A_N) - Y)_F^2$ be the objective function we wish to minimize in @eq-constrained-least-squares. Following Xu and Yin @xu_BlockCoordinateDescent_2013, the general approach we take to minimize $f$ is to apply block coordinate descent using each factor as a different block. Let $A_n^t$ be the $t$th iteration of the $n$th factor, and let

$ f_n^t (A_n) := 1 / 2 norm(g (A_1^(t + 1) , dots.h , A_(n - 1)^(t + 1) , A_n , A_(n + 1)^t , dots.h , A_N^t) - Y)_F^2 $

be the (partially updated) objective function at iteration $t$ for factor $n$.

Given initial factors $A_1^0 , dots.h , A_N^0$, we cycle through the factors $n in [N]$ and perform the update

$ A_n^(t + 1) arrow.l arg min_(A_n in cal(C)_n) ⟨nabla f_n^t (A_n^t) , A_n - A_n^t⟩ + L_n^t / 2 norm(A_n - A_n^t)_F^2 , $

for $t = 1 , 2 , dots.h$ until some convergence criterion is satisfied (see @sec-convergence-criteria).

This implicit update has the #emph[projected gradient descent] closed form solution for convex constraints $cal(C)_n$,

#math.equation(block: true, numbering: "(1)", [ $ A_n^(t + 1) arrow.l P_(cal(C)_n) (A_n^t - 1 / L_n^t nabla f_n^t (A_n^t)) . $ ])<eq-proximal-explicit>

We typically choose $L_n^t$ to be the Lipschitz constant of $nabla f_n^t$, since it is a sufficient condition to guarantee $f_n^t (A_n^(t + 1)) lt.eq f_n^t (A_n^t)$, but other step sizes can be used in theory @nesterov_NonlinearOptimization_2018[Sec. 1.2.3].

?ASIDE? To write $nabla f_n^t$, we have assumed (block) differentiability of the decomposition model $g$. In practice, most decompositions are "block-linear" (freeze all factors but one and you have a linear function) and in rare cases are "block-affine". "block-affine" is enough to ensure $f_n^t$ is convex (i.e.~$f$ is "block-convex") so the updates @eq-proximal-explicit converge to a Nash equilibrium (block minimizer).

=== High level code
<high-level-code>
To ensure the code stays flexible, the main algorithm of BlockTensorDecomposition.jl, `factorize`, is defined at a very high level.

```julia
factorize(Y; kwargs...) =
    _factorize(Y; (default_kwargs(Y; kwargs...))...)

"""
Inner level function once keyword arguments are set
"""
function _factorize(Y; kwargs...)
    decomposition, previous, updateprevious!, parameters, updateparameters!,
    update!, stats_data, getstats, converged, kwargs = initialize(Y, kwargs)

    while !converged(stats_data; kwargs...)
        # Usually one cycle of updates through each factor in the decomposition
        update!(decomposition; parameters...)

        # This could be the next stepsize or other info used by update!
        updateparameters!(parameters, decomposition, previous)

        push!(stats_data,
            getstats(decomposition, Y, previous, parameters, stats_data))

        # Update one or two previous iterates. For example, used for momentum
        updateprevious!(previous, parameters, decomposition)
    end

    kwargs = postprocess!(decomposition, Y, previous, parameters, stats_data, updateparameters!, getstats, kwargs)

    return decomposition, stats_data, kwargs
end
```

The magic of the code is in defining the functions at runtime for a particular decomposition requested, from a reasonable set of default keyword arguments. This is discussed further in @sec-flexibility.

=== Computing Gradients
<sec-gradient-computation>
- Use Auto diff generally
- But hand-crafted gradients and Lipschitz calculations #emph[can] be faster (e.g.~symmetrized slice-wise dot product)

Generally, we can use automatic differentiation on $f$ to compute gradients. Some care needs to be taken otherwise the forward or backwards pass will have to be recompiled every iteration since the factors are updated every iteration.

But for Tucker decompositions, we can compute gradients faster than what an automatic differentiation scheme would give, by taking advantage of symmetry and other computational shortcuts.

Starting with the Tucker-1 decomposition (@def-tucker-1-decomposition), we would like to compute $nabla_B f (B , A)$ and $nabla_A f (B , A)$ for $f (B , A) = 1 / 2 norm(A B - Y)_F^2$ for a given input $Y$. We have the gradient

#math.equation(block: true, numbering: "(1)", [ $ nabla_B f (B , A) = A^tack.b (A B - Y) = (B times_1 A - Y) times_1 A^tack.b $ ])<eq-tucker-1-gradient-1>

by chain rule, but it is more efficient to calculate the gradient as

#math.equation(block: true, numbering: "(1)", [ $ nabla_B f (B , A) = (A^tack.b A) B - A^tack.b Y = B times_1 (A^tack.b A) - Y times_1 A^tack.b . $ ])<eq-tucker-1-gradient-2>

#footnote[Seeing @eq-tucker-1-gradient-1 and @eq-tucker-1-gradient-2 written using the $1$-mode product shows how it is "backwards" to normal matrix-matrix multiplication.];For $A in bb(R)^(I times R)$, $B in bb(R)^(R times J times K)$, and $Y in bb(R)^(I times J times K)$, @eq-tucker-1-gradient-1 requires

$ underbrace(2 I J K R, A B - Y) + underbrace(I J K (2 I - 1), A^tack.b (A B - Y)) tilde.op 2 I J K R + 2 I^2 J K $ floating point operations (FLOPS) whereas @eq-tucker-1-gradient-2 only uses

$ underbrace(frac(R (R + 1), 2) (2 I - 1), A^tack.b A) + underbrace(R J K (2 I - 1), A^tack.b Y) + underbrace(2 R^2 J K, (A^tack.b A) B - (A^tack.b Y)) tilde.op 2 I J K R + 2 R^2 J K + I R^2 $

FLOPS#footnote[Note we have the smaller factor $R (R + 1) \/ 2$ and not the expected $R^2$ number of entries needed to compute $A^tack.b A$. The product is a symmetric matrix so only the upper or lower triangle of entries needs to be computed.]. So for small ranks $R lt.double I$, @eq-tucker-1-gradient-2 is cheaper.

A similar story can be said about $nabla_A f (B , A)$ which is most efficiently computed as

$ nabla_A f (B , A) = A (B dot.op_1 B) - Y dot.op_1 B . $

#block[
#callout(
body: 
[
For the family of Tucker decompositions, the objective function $f$ is "block-quadratic" with respect to the factors. This means the gradient with respect to a factor is an affine function of that factor. This is exactly what we see in @eq-tucker-1-gradient-2 where $B$ is multiplied by the "slope" $A^tack.b A$ plus a shift of $- Y times_1 A^tack.b$.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
)
]
The associated implementation with BlockTensorDecomposition.jl is shown below. We define a `make_gradient` which takes the decomposition, factor index `n`, and data tensor `Y`, and creates a function that computes the gradient for the same type of decomposition. This lets us manipulate the function that computes the gradient, rather than just the computed gradient.

```julia
function make_gradient(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function gradient0(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            AA = A'A
            YA = Y×₁A'
            grad = B×₁AA - YA
            return grad
        end
        return gradient0
    elseif n==1 # the matrix is the first factor
        function gradient1(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            BB = slicewise_dot(B, B)
            YB = slicewise_dot(Y, B)
            grad = A*BB - YB
            return grad
        end
        return gradient1
    else
        error("No $(n)th factor in Tucker1")
    end
end
```

Similarly, we also have special methods for the Tucker and CP Decomposition.

The gradient with respect to the core for a full Tucker factorization is

$ nabla_B f (B , A_1 , dots.h , A_N) = B times.big_n A_n^tack.b A_n - Y times.big_n A_n^tack.b , $

and the gradient with respect to the matrix factor $A_n$ is

$ nabla_(A_n) f (B , A_1 , dots.h , A_N) = A_n (tilde(X)_n dot.op_n tilde(X)_n) - Y dot.op_n tilde(X)_n $

where

$ tilde(X)_n = (B times.big_(m eq.not n) A_m) = lr(bracket.l.double B \; A_1 , dots.h , A_(n - 1) , upright("id")_(R_n) , A_(n + 1) , dots.h , A_N bracket.r.double) . $

```julia
function make_gradient(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function gradient_core(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            gram_matrices = map(A -> A'A, matrices) # gram matrices AA = A'A,
                                                    # BB = B'B, ...
            grad = tuckerproduct(B, gram_matrices)
                 - tuckerproduct(Y, adjoint.(matrices))
            return grad
        end
        return gradient_core

    elseif n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

For the CP Decomposition, we can simply treat the core as $B = upright("id")_R$ and compute the gradient with respect to the matrix factors similarly to the Tucker decomposition:

$ nabla_(A_n) f (A_1 , dots.h , A_N) = A_n (tilde(X)_n dot.op_n tilde(X)_n) - Y dot.op_n tilde(X)_n $

where

$ tilde(X)_n = (upright("id")_R times.big_(m eq.not n) A_m) = lr(bracket.l.double upright("id")_R \; A_1 , dots.h , A_(n - 1) , upright("id")_R , A_(n + 1) , dots.h , A_N bracket.r.double) . $

```julia
function make_gradient(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

=== Computing Lipschitz Step-sizes
<sec-lipschitz-computation>
Similar to automatic differentiation, there exist "automatic Lipschitz" calculations to upper bound the Lipschitz constant of a function @virmaux_lipschitz_2018.

For the family of Tucker decompositions, we can compute the Lipschitz constants of the gradient efficiently similar to how we compute the gradient in @sec-gradient-computation with the following corollaries of @thm-quadratic-smoothness.

#corollary()[
Let $B in bb(R)^(R_1 times dots.h.c times R_N)$, $A_m in bb(R)^(I_m times R_m)$, and $Y in bb(R)^(I_1 times dots.h.c times I_N)$. The function

$ f (A) = 1 / 2 norm([B \; A_1 , dots.h , A_(n - 1) , A , A_(n + 1) , dots.h , A_N] - Y)_F^2 $

is quadratic, and $L$-smooth with constant

$ L_(A_n) = norm(tilde(X)_n dot.op_n tilde(X)_n)_(upright("op")) $

where

$ tilde(X)_n = (B times.big_(m eq.not n) A_m) = lr(bracket.l.double B \; A_1 , dots.h , A_(n - 1) , upright("id")_(R_n) , A_(n + 1) , dots.h , A_N bracket.r.double) . $

#block[
#emph[Proof]. The result follows from @thm-operator-norm-outer-product and @thm-quadratic-smoothness.

]
] <cor-least-squares-matrix>
#corollary()[
Let $A_n in bb(R)^(I_n times R_n)$, and $Y in bb(R)^(I_1 times dots.h.c times I_N)$. The function

$ f (B) = 1 / 2 norm([B \; A_1 , dots.h , A_N] - Y)_F^2 $

is quadratic, and $L$-smooth with constant

$ L_B = product_(n = 1)^N norm(A_n^tack.b A_n)_(upright("op")) . $

#block[
#emph[Proof]. The result follows from @thm-operator-norm-outer-product and @thm-quadratic-smoothness.

]
] <cor-least-squares-core>
This yields the following efficient implementations.

#block[
#callout(
body: 
[
It is tempting to use the identity $norm(A^tack.b A)_(upright("op")) = norm(A)_(upright("op"))^2$ to calculate the Lipschitz constant without forming $A^tack.b A$. For tall dense matrices, using this identity is slower and more memory intensive as of Julia 1.11.2. See #link("https://github.com/JuliaLang/LinearAlgebra.jl/issues/1185")[LinearAlgebra.jl issue 1185] on Github.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
)
]
```julia
function make_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(X::AbstractTucker; kwargs...)
            return prod(A -> opnorm(A'A), matrix_factors(X))
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            return opnorm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

In the case of Tucker decomposition, the Lipschitz constants simplify to

$ L_B = norm(A^tack.b A)_(upright("op")) , quad L_B = norm(B dot.op_1 B)_(upright("op")) . $

```julia
function make_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(X::Tucker1; kwargs...)
            A = matrix_factor(X, 1)
            return opnorm(A'A)
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(X::Tucker1; kwargs...)
            B = core(X)
            return opnorm(slicewise_dot(B, B))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end
```

Lastly, for CP decomposition, the Lipschitz constants for the matrices can be calculated similarly to the Tucker decomposition.

```julia
function make_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N
        function lipschitz_matrix(X::CPDecomposition; kwargs...)
            id = core(X)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(id, matrices; exclude=n)
            return opnorm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in CPDecomposition")
    end
end
```

= Computational Techniques
<computational-techniques>
- As stated, algorithm works
- But can be slow, especially for constrained or large problems

As stated, the algorithm described in @sec-base-algorithm works. It will converge to a solution to our optimization problem and factorize the input tensor. It is worth discussing how the algorithm can be modified to improve convergence to maintain quick convergence for large problems, and what sort of architectural methods are used to allow for maximum flexibility, without over engineering the package.

== For Improving Convergence Speed
<for-improving-convergence-speed>
There are a few techniques used to assist convergence. Two ideas that are well studied are discussed in this section. They are 1) breaking up the updates into smaller blocks, and 2) using momentum or acceleration. What is perhaps novel is considering the synergy between these two ideas.

Two more techniques are implemented in BlockTensorDecomposition.jl to improve convergence. To the authors knowledge, these are new to tensor factorization, but may or may not be applicable depending on the exact factorization problem or data being studied. For these reasons, these other techniques are discussed separately in @sec-ppr and @sec-multi-scale.

=== Sub-block Descent
<sub-block-descent>
- Use smaller blocks, but descent in parallel (sub-blocks don’t wait for other sub-blocks)
- Can perform this efficiently with a "matrix step-size"

When using block coordinate descent as in @sec-base-algorithm, it is natural to treat each factor as its own block. This requires the fewest blocks while ensuring the objective is still convex with respect to each block. We could just as easily use smaller blocks.

In the case of Tucker decomposition, one modification of the update shown in @eq-proximal-explicit would be to update each column $a_(n , r)$, $r = 1 , dots.h , R_n$ of the matrix $A_n$ separately. This would be suitable if the constraint that $A_N in cal(C)_n$ can be broken up further into the constraints $a_(n , r) in cal(C)_(n , r)$. This is shown in the following update scheme:

#math.equation(block: true, numbering: "(1)", [ $ a_(n , r)^(t + 1) arrow.l P_(cal(C)_(n , r)) (a_(n , r)^t - 1 / L_(n , r)^t nabla f_(n , r)^t (a_(n , r)^t)) , $ ])<eq-sub-block-update-proper>

where $f_(n , r)^t (a) = 1 / 2 norm([B \; A_1^(t + 1) , dots.h , A_(n - 1)^(t + 1) , A_(n , r)^t (a) , A_(n + 1)^t , dots.h , A_N^t] - Y)_F^2$ and

#math.equation(block: true, numbering: "(1)", [ $ A_(n , r)^t (a) = mat(delim: "[", arrow.t, , arrow.t, arrow.t, arrow.t, , arrow.t; a_(n , 1)^(t + 1), dots.h.c, a_(n , r - 1)^(t + 1), a, a_(n , r + 1)^t, dots.h.c, a_(n , R_n)^t; arrow.b, , arrow.b, arrow.b, arrow.b, , arrow.b; #none) . $ ])<eq-proper-column-update>

In theory, the block update shown in @eq-sub-block-update-proper should be a bit more expensive than using the larger blocks on the matrices $A$ shown in @eq-proximal-explicit, since the gradient needs to be recomputed $R_n$ times for each matrix block $n$, rather than only computing the gradient once per block $n$. To get around this, we use the fact that $nabla f_(n , r)^t (a)$ is the $r$th column from the gradient $nabla f_n^t (A)$ where $f_n^t (A) = 1 / 2 norm([B \; A_1^(t + 1) , dots.h , A_(n - 1)^(t + 1) , A , A_(n + 1)^t , dots.h , A_N^t] - Y)_F^2$. So we can approximate @eq-sub-block-update-proper by first calculating the gradient $nabla f_n^t$ at

#math.equation(block: true, numbering: "(1)", [ $ hat(A)_n^t = mat(delim: "[", arrow.t, , arrow.t, arrow.t, arrow.t, , arrow.t; a_(n , 1)^t, dots.h.c, a_(n , r - 1)^t, a_(n , r)^t, a_(n , r + 1)^t, dots.h.c, a_(n , R_n)^t; arrow.b, , arrow.b, arrow.b, arrow.b, , arrow.b; #none) , $ ])<eq-merged-column-update>

and then updating each sub-block $r$ according to

#math.equation(block: true, numbering: "(1)", [ $ a_(n , r)^(t + 1) arrow.l P_(cal(C)_(n , r)) (a_(n , r)^t - 1 / L_(n , r)^t nabla f_n^t (hat(A)_n^t)) . $ ])<eq-sub-block-update-half-merged>

Note the difference between @eq-proper-column-update and @eq-merged-column-update is that we don’t use the most recent columns $a_(n , j)$ for $j < r$ in @eq-merged-column-update.

The update given in @eq-sub-block-update-half-merged can be merged back to an update on the whole block $A_n$

#math.equation(block: true, numbering: "(1)", [ $ A_n^(t + 1) arrow.l P_(cal(C)_n) (A_n^t - nabla f_n^t (A_n^t) (hat(L)_n^t)^(- 1)) $ ])<eq-sub-block-update>

where we have the $R_n times R_n$ diagonal "Lipschitz Matrix"

$ hat(L)_n^t = mat(delim: "[", L_(n , 1)^t, 0, , 0; 0, L_(n , 2)^t, , 0; #none, , dots.down, dots.v; 0, 0, dots.h.c, L_(n , R_n)^t) . $

It is not too hard to show that the Lipschitz $L_(n , r)^t$ for $nabla f_(n , r)^t$ is the Euclidean norm of the $r$th column#footnote[We could have used the $r$th row of $tilde(X)_n dot.op_n tilde(X)_n$ since this matrix is symmetric. Since Julia store matrices in column-major order, many operations that perform column-wise are more efficient than their equivalent row-wise operation.] of the matrix $tilde(X)_n dot.op_n tilde(X)_n$ from @cor-least-squares-matrix,

$ L_(n , r)^t = norm((tilde(X)_n dot.op_n tilde(X)_n) [: , r])_2 . $

This leads to the following efficient calculation of the Lipschitz matrix in Julia.

```julia
function diagonal_lipschitz_matrix(T::Tucker, n::Int; kwargs...)
    B = core(T)
    matrices = matrix_factors(T)
    X̃ₙ = tuckerproduct(B, matrices; exclude=n)
    return Diagonal_col_norm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
end

Diagonal_col_norm(X) = Diagonal(norm.(eachcol(X)))
```

We can now compare the merged sub-block update @eq-sub-block-update to the standard projected gradient descent update shown in @eq-proximal-explicit. The difference is that we calculate a "matrix step-size" $hat(L)_n^t in bb(R)^(R_n times R_n)$ rather than a scalar $L_n^t in bb(R)$. In practice, this leads to an improvement in convergence speed for two reasons.

First, computing the matrix $hat(L)_n^t$ often faster than the scalar $L_n^t = norm(tilde(X)_n dot.op_n tilde(X)_n)_(upright("op"))$. The former only requires calculating the Euclidean norm of $R$ vectors for a total cost of $2 R_n^2$ floating point operations (FLOPs), whereas the latter requires the top eigenvalue of $tilde(X)_n dot.op_n tilde(X)_n$. This is usually done with a power method or truncated SVD which can be costlier than the flat rate of $2 R_n^2$ FLOPs.

Secondly, using the matrix $hat(L)_n^t$ means columns where $L_(n , r)^t$ is small can take larger descent steps. This is because the largest singular value of $tilde(X)_n dot.op_n tilde(X)_n$ is an upper bound on the Euclidean norm of each column: $L_(n , r)^t lt.eq L_n^t$. Using the scaler Lipschitz $L_n^t$ is equivalent to the diagonal matrix

$ D = mat(delim: "[", L_n^t, 0, , 0; 0, L_n^t, , 0; #none, , dots.down, dots.v; 0, 0, dots.h.c, L_n^t) $

in the merged sub-block update shown in @eq-sub-block-update. So each column of $A_n$ is forced to use the worst case (largest) singular value of $tilde(X)_n dot.op_n tilde(X)_n$. In this way, the matrix $hat(L)_n^t$ acts like a cheap approximate Hessian as if we were doing a quasi-Newton update with step-size $1$.

For completeness, we can perform the same merged sub-block update to update the core $B$. In this case, we obtain the more complicated "Lipschitz tensor" $hat(L)_B^t in bb(R)^((R_1 times dots.h.c times R_N)^2)$ defined by

$ hat(L)_B^t = hat(L)_(B , 1)^t times.circle dots.h.c times.circle hat(L)_(B , N)^t $

where each matrix $hat(L)_(B , n)^t in bb(R)^(R_n times R_n)$ is diagonal with non-zero entries

$ L_(B , n)^t [r , r] = norm(((A_n^t)^tack.b A_n^t) [: , r])_2 . $

The merged sub-block update for the core becomes

#math.equation(block: true, numbering: "(1)", [ $ B^(t + 1) arrow.l P_(cal(C)_B) (B^t - nabla f_0^t (B^t) times_B (hat(L)_B^t)^(- 1)) $ ])<eq-sub-block-update-core>

with the multiplication

#math.equation(block: true, numbering: "(1)", [ $ nabla f_0^t (B^t) times_B (hat(L)_B^t)^(- 1) & = nabla f_0^t (B^t) times.big_n (hat(L)_(B , n)^t)^(- 1)\
 & = lr(bracket.l.double nabla f_0^t (B^t) \; (hat(L)_(B , 1)^t)^(- 1) , dots.h , (hat(L)_(B , n)^t)^(- 1) bracket.r.double) . $ ])<eq-tensor-matrices-product>

This should be thought of as normalizing each dimension of the tensor $nabla f_0^t (B^t)$ so that we can take a unit step-size.

TODO use the multi-mode product in stead of defining a new multiplication $times_B$ here. I think I’ll have the same tensor product issue as before where the indices.

Putting the core and matrices Lipschitz calculations together gives us the following Julia code. Note we store $hat(L)_B^t$ in factored form as a tuple of diagonal matrices to save space and computation.

```julia
function make_block_lipschitz(T::AbstractTucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(X::AbstractTucker; kwargs...)
            return map(A -> Diagonal_col_norm(A'A), matrix_factors(X))
        end # Returns a tuple of diagonal matrices
        return lipschitz_core

    elseif n in 1:N
        function lipschitz_matrix(X::AbstractTucker; kwargs...)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(core(X), matrices; exclude=n)
            return Diagonal_col_norm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

=== Momentum
<momentum>
- This one is standard
- Use something similar to @xu_BlockCoordinateDescent_2013
- This is compatible with sub-block descent with appropriately defined matrix operations

In practice, we find that extrapolating the iterate based on the prior iterate

#math.equation(block: true, numbering: "(1)", [ $ hat(A)_n^t arrow.l A_n^t + omega_n^t (A_n^t - A_n^(t - 1)) $ ])<eq-extrapolation-ideal>

for some amount of extrapolation $omega_n^t gt.eq 0$ before applying the update @eq-sub-block-update greatly improves the speed of descent. This can be thought of as a type of momentum where we continue to move in directions that showed a lot of improvement during the last iteration.

Our selection for $omega_n^t$ follows Xu and Yin’s method for block coordinate descent @xu_BlockCoordinateDescent_2013, which is itself inspired by Tseng and Yun’s coordinate gradient descent method @tseng_coordinate_2009.

Given a parameter#footnote[Usually we pick a number close to $1$. For example, we use the default $delta = 0.9999$.] $delta in \[ 0 , 1 \)$, we define the momentum parameters and $tau^t$ and $omega_n^t$ according to the following updates

#math.equation(block: true, numbering: "(1)", [ $ tau^0 & = 1\
tau^(t + 1) & arrow.l 1 / 2 (1 + sqrt(1 + 4 (tau^t)^2))\
hat(omega)^t & arrow.l frac(tau^t - 1, tau^(t + 1))\
omega_n^t & arrow.l min (hat(omega)^t , delta sqrt(hat(L)_n^(t - 1) (hat(L)_n^t)^(- 1))) . $ ])<eq-momentum-parameters>

TODO notation is going to get confusing. We use hat/not hat $L$ for the scaler vs matrix/tensor version. But we use hat/not hat $omega$ for the ideal vs clamped momentum.

What is novel with our approach is that we perform this momentum on the Lipschitz matrices and tensors $hat(L)_n^t$ rather than scaler Lipschitz constant $L_n^t$. In this way, we should interpret the operations shown in @eq-momentum-parameters as operating element-wise. This also means the momentum parameter $omega_n^t$ is a matrix or tensor and takes the same shape as $hat(L)_n^t$.

In order to perform @eq-extrapolation-ideal, we use the equivalent but more efficient formulation

$ hat(A)_n^t & arrow.l A_n^t (upright("id")_(R_n) + omega_n^t) - A_n^(t - 1) omega_n^t . $

```julia
function (U::MomentumUpdate)(X::T; X_last::T, ω, δ, kwargs...) where T
    n = U.n

    L = U.lipschitz(X; kwargs...)
    L_last = U.lipschitz(X_last; kwargs...)
    ω = min.(ω, δ .* .√(L_last/L))

    A, A_last = factor(X, n), factor(X_last, n)

    A .= U.combine(A, id + ω)
    A .-= U.combine(A_last, ω)
end
```

In the code above, the momentum stores the factor $n$ in acts on, how to compute the Lipschitz constant, matrix, or tensor, and how to combine (multiply) the constant with the factor. In the case of matrix factors $A_n$ in a Tucker decomposition, this is simply right matrix-matrix multiplication. The core factor $B$ uses $times_B$ as described in @eq-tensor-matrices-product.

The parameters of the momentum update are handled separately. This is to treat the momentum update as "apply this update with these parameters". The parameters $tau^t$ and $hat(omega)^t$ are updated by the following function that keeps track of all parameters needed to perform the iteration. In this case, we keep track of what iteration $t$ we are at, the previous iterate, and a few options for the order in which to cycle through and update the blocks.

```julia
update_τ(τ) = 0.5*(1 + sqrt(1 + 4*τ^2))

function initialize_parameters(decomposition, Y, previous; momentum::Bool, random_order, recursive_random_order, kwargs...)
    # parameters for the update step are symbol => value pairs
    # held in a dictionary since we may mutate these, e.g. the step-size
    parameters = Dict{Symbol, Any}()

    # General Looping
    parameters[:iteration] = 0
    parameters[:X_last] = previous[begin] # Last iterate
    parameters[:random_order] = random_order
    parameters[:recursive_random_order] = recursive_random_order

    # Momentum
    if momentum
        parameters[:τ_last] = float(1) # need this field to hold Floats, not Ints
        parameters[:τ] = update_τ(float(1))
        parameters[:ω] = (parameters[:τ_last] - 1) / parameters[:τ]
        parameters[:δ] = kwargs[:δ]
    end

    function updateparameters!(parameters, decomposition, previous)
        parameters[:iteration] += 1
        # parameters[:x_last] = previous[begin]
# This is commented since parameters[:x_last] already points to previous[begin]

        if momentum
            parameters[:τ_last] = parameters[:τ]
            parameters[:τ] = update_τ(parameters[:τ_last])
            parameters[:ω] = (parameters[:τ_last] - 1) / parameters[:τ]
        end
    end

    return parameters, updateparameters!
end
```

=== Empirical Evidence for Sub-Block Descent and Momentum
<empirical-evidence-for-sub-block-descent-and-momentum>
To showcase that the combination of these two tricks can speed up convergence, we will benchmark them by factorizing a random $10 times 10$ tensor (a matrix) with rank $3$. The Julia code is shown below, and the results are shown in @tbl-subblock-momentum-results.

```julia
using BenchmarkTools
using BlockTensorDecomposition

fact = BlockTensorDecomposition.factorize

options = (
    :rank => 3,
    :tolerence => (1, 0.03),
    :converged => (GradientNNCone, RelativeError),
    :δ => 0.9,
)

n_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=false,
    options...
)

y_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=false,
    options...
)

n_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=true,
    options...
)

y_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=true,
    options...
)

I, J = 10, 10
R = 3

@benchmark n_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark n_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark y_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark y_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))

performance_increase(old, new) = (old - new) / new * 100
```

The code `Tucker1((I, J), R)` produces a random $I times J$ rank-$R$ matrix by generating two matrices $A in bb(R)^(I times R)$ and $B in bb(R)^(R times J)$ with standard normal entries, and multiplies them together.

#figure([
#table(
  columns: (25.33%, 37.33%, 37.33%),
  align: (auto,auto,auto,),
  table.header([], [#strong[No Momentum];], [#strong[Yes Momentum];],),
  table.hline(),
  [#strong[No Sub-Block];], [48.843 ms], [45.738 ms (6.7887% faster)],
  [#strong[Yes Sub-Block];], [27.473 ms (77.785% faster)], [#strong[24.350 ms (100.59% faster)];],
)
], caption: figure.caption(
position: top, 
[
Summary of median times to factorize a random $10 times 10$ rank-3 matrix under different methods. The performance increase is given by the formula $(upright("old") - upright("new")) \/ upright("new")$.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-subblock-momentum-results>


In @tbl-subblock-momentum-results, you can see that having both sub-block descent and momentum yields the fastest factorization. Moreover, the performance increase is #emph[more] than simply the performance increases obtained by exclusively sub-block or momentum alone.#footnote[The expected performance increase if sub-block descent and momentum where independence would be $(1 + 0.067887) (1 + 0.77785) = 1.89854$ or only 89.854% faster.] This suggests that there is synergy with these two methods and are best used together.

== For Flexibility
<sec-flexibility>
- there are a number of software engineering techniques used
- these help flexibility for hot swapping and a language for making custom…
  - convergence criterion (and having multiple stopping conditions)
  - probing info during the iterations (stats collected at the end)
  - having multiple constraints and ways to enforce them
  - cyclically or partially randomly or fully randomly update factors
- smart enough to apply these in a reasonable order

=== Convergence Criteria and Stats
<sec-convergence-criteria>
- Can request info about any factor at each outer iteration
- any subset of stats can be the convergence criteria

=== `BlockUpdate` Language
<blockupdate-language>
- construct the updates as a list of updates
- very functional programming
- can apply them in sequence or in a random order (or partially random)

=== Constraints
<constraints>
- one type of update (other than the typical GD update)
- can combine them with composition
  - which is different than projecting onto their intersection!
- Constraint updates combine the constraint with how they are enforced
  - need to go together since there are multiple ways to enforce them e.g.~simplex (see next section)

= Partial Projection and Rescaling
<sec-ppr>
- for bounded linear constraints
  - first project
  - then rescale to enforce linear constraints
- faster to execute then a projection
- often does not loose progress because of the rescaling (decomposition dependent)

= Multi-scale
<sec-multi-scale>
- use a coarse discretization along continuous dimensions
- factorize
- linearly interpolate decomposition to warm start larger decompositions

= Conclusion
<conclusion>
- all-in-one package
- provide a playground to invent new decompositions
- like auto-diff for factorizations

= Appendix
<appendix>
== Building the Hessian from two gradients
<sec-hessian-from-gradient>
To build the Hessian from the definition of the gradient, we first extend the gradient to tensor-valued functions. For a function $F : bb(R)^(J_1 times dots.h.c times J_N) arrow.r bb(R)^(I_1 times dots.h.c times I_M)$ where

$ F (X) = [f_(i_1 dots.h i_M) (X)] $

is a tensor of scalar functions $f_(i_1 dots.h i_M) : T arrow.r bb(R)$, the gradient of $F$ at $X$ is defined entry-wise as

$ nabla F (X) [i_1 , dots.h , i_M] [j_1 , dots.h , j_N] & = nabla f_(i_1 dots.h i_M) (X) [j_1 , dots.h , j_N]\
 & = frac(partial f_(i_1 dots.h i_M), partial X [j_1 , dots.h , j_N]) (X) . $

This treats the gradient at $X$ as a tensor of tensors $nabla F (X) in (bb(R)^(I_1 times dots.h.c times I_M))^(J_1 times dots.h.c times J_N)$. This is naturally isomorphic to a tensor of order $M + N$ with entries

#math.equation(block: true, numbering: "(1)", [ $ nabla F (X) [i_1 , dots.h , i_M , j_1 , dots.h , j_N] = frac(partial f_(i_1 dots.h i_M), partial X [j_1 , dots.h , j_N]) (X) . $ ])<eq-tensor-gradient-entries>

So we conclude that the gradient at $X$ of a tensor-valued function $F$ is $nabla F : bb(R)^(J_1 times dots.h.c times J_N) arrow.r bb(R)^(I_1 times dots.h.c times I_M times J_1 times dots.h.c times J_N)$ is given by @eq-tensor-gradient-entries.

We can define the Hessian of a scalar function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ at $X$ as $nabla^2 f (X) = nabla (nabla f) (X) : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^((I_1 times dots.h.c times I_N)^2)$. The inner nabla $nabla$ is the gradient of the scalar function $f$, and the outer nabla $nabla$ is the gradient of the tensor-valued function $nabla f$.

This means

$ nabla^2 f (A) [i_1 , dots.h , i_N , j_1 , dots.h , j_N] = frac(partial^2 f, partial A [j_1 , dots.h , j_N] partial A [i_1 , dots.h , i_N]) (A) , $

but if the function has continuous second derivatives, we can perform the partial derivatives in either order

$ frac(partial^2 f, partial A [j_1 , dots.h , j_N] partial A [i_1 , dots.h , i_N]) (A) = frac(partial^2 f, partial A [i_1 , dots.h , i_N] partial A [j_1 , dots.h , j_N]) (A) . $

 
  
#set bibliography(style: "citationstyles/ieee-compressed-in-text-citations.csl") 


#bibliography("references.bib")

