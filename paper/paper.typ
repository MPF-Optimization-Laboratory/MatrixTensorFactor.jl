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

We use $[N] = { 1 , 2 , dots.h , N } = { n }_(n = 1)^N$ to denote integers from $1$ to $N$.

Usually, lower case symbols will be used for the running index, and the capitalized letter will be the maximum letter it runs to. This leads to the convenient shorthand $i in [I]$, $j in [J]$, etc.

The set of nonnegative numbers is denoted as $bb(R)_(+) = bb(R)_(gt.eq 0) = {x in bb(R) mid(bar.v) x gt.eq 0}$.

Vectors are denoted with lowercase letters ($x$, $y$, etc.), and matrices and higher order tensors with uppercase letters (commonly $A$, $B$, $C$ and $X$, $Y$, $Z$). The order of a tensor is the number of axes it has. We would call vectors "order-1" or "1st order" tensors, and matrices "order-2" or "2nd order" tensors.

To avoid confusion between entries of a vector/matrix/tensor and indexing a list of objects, we use square brackets to denote the former, and subscripts to denote the later. For example, the entry in the $i$th row and $j$th column of a matrix $A in bb(R)$ is $A [i , j]$. This follows MATLAB/Julia notation where `A[i,j]` points to the entry $A [i , j]$. We contrast this with a list of $I$ objects being denoted as $a_1 , dots.h , a_I$, or more compactly, ${ a_i }$ when it is clear the index $i in [I]$.

The $n$-fibres, $n$th mode fibres, or mode $n$ fibres of an $N$th order tensor $A$ are denoted $A [i_1 , med dots.h , med i_(n - 1) , med : , med i_(n + 1) , med dots.h , med i_N]$. For example, the 1-fibres of a matrix $M$ are the column vectors \
$M [: , med j]$, and the 2-fibres are the row vectors $M [i , med :]$. For order-3 tensors, the $1$st, $2$nd, and $3$rd mode fibres $A [: , j , k]$, $A [i , : , :]$, and $A [i , j , :]$ are called the vertical/column, horizontal/row, and depth/tube fibres respectively and are displayed in @fig-tensor-fibres. In Julia, the 1-, 2-, and 3-fibres of a third order array `A` would be `eachslice(A, dims=(2,3))`, `eachslice(A, dims=(1,3))`, and `eachslice(A, dims=(1,2))`.

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


The $n$-slices, $n$th mode slices, or mode $n$ slices of an $N$th order tensor $A$ are notated with the slice $A [: , med dots.h , med : , med i_n , med : , med dots.h , med :]$. For matrices, the 1-fibres are the same as the 2-slices (and vice versa), but for $N$th order tensors in general, fibres are always vectors, whereas $n$-slices are $(N - 1)$th order tensors. For a $3$rd order tensor $A$, the $1$st, $2$nd, and $3$rd mode slices $A [i , : , :]$, $A [: , j , :]$, and $A [: , : , k]$ have special names and are called the horizontal, lateral, and frontal slices and are displayed in @fig-tensor-slices. In Julia, the 1-, 2-, and 3-slices of a third order array `A` would be `eachslice(A, dims=1)`, `eachslice(A, dims=2)`, and `eachslice(A, dims=3)`.

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


The Frobenius inner product between two tensors $A , B in bb(R)^(I_1 times dots.h times I_N)$ is denoted

$ ⟨A , B⟩ = A dot.op B = sum_(i_1 = 1)^(I_1) dots.h sum_(i_N = 1)^(I_N) A [i_1 , dots.h , i_N] B [i_1 , dots.h , i_N] . $

Since we commonly use $I$ as the size of a tensor’s dimension, we use $upright(i d)_I$ to denote the identity tensor of size $I$ (of the appropriate order). When the order is $2$, $upright(i d)_I$ is an $I times I$ matrix with ones along the main diagonal, and zeros elsewhere. For higher orders $N$, this is an $underbrace(I times dots.h.c times I, N upright("times"))$ tensor where $upright(i d)_I [i_1 , dots.h , i_N] = 1$ when $i_1 = dots.h = i_N in [I]$, and is zero otherwise.

== Common Decompositions
<common-decompositions>
- Extensions of PCA/ICA/NMF to higher dimensions
- talk about the most popular Tucker, Tucker-n, CP
- other decompositions
  - high order SVD (see Kolda and Bader)
  - HOSVD (see Kolda, Shifted power method for computing tensor eigenpairs)

== Tensor rank
<tensor-rank>
- tensor rank
- constrained rank (nonnegative etc.)

= Computing Decompositions
<computing-decompositions>
- Given a data tensor and a model, how do we fit the model?

== Optimization Problem
<optimization-problem>
- Least squares (can use KL, 1 norm, etc.)

== Base algorithm
<base-algorithm>
- Use Block Coordinate Descent / Alternating Proximal Descent
  - do #emph[not] use alternating least squares (slower for unconstrained problems, no closed form update for general constrained problems)

= Techniques for speeding up convergences
<techniques-for-speeding-up-convergences>
- As stated, algorithm works
- But can be slow, especially for constrained or large problems

== Sub-block Descent
<sub-block-descent>
- Use smaller blocks, but descent in parallel (sub-blocks don’t wait for other sub-blocks)
- Can perform this efficiently with a "matrix step-size"

== Momentum
<momentum>
- This one is standard
- Use something similar to @xu_BlockCoordinateDescent_2013
- This is compatible with sub-block descent with appropriately defined matrix operations

== Partial Projection and Rescaling
<partial-projection-and-rescaling>
- for bounded linear constraints
  - first project
  - then rescale to enforce linear constraints
- faster to execute then a projection
- often does not loose progress because of the rescaling (decomposition dependent)

== Multi-scale
<multi-scale>
- use a coarse discretization along continuous dimensions
- factorize
- linearly interpolate decomposition to warm start larger decompositions

= Conclusion
<conclusion>
- all-in-one package
- provide a playground to invent new decompositions
- like auto-diff for factorizations

 
  
#set bibliography(style: "citationstyles/ieee-compressed-in-text-citations.csl") 


#bibliography("references.bib")

