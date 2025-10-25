#!/usr/bin/env python3
"""
Script to render TikZ diagram to SVG/PNG for GitHub display
"""

import subprocess
import os
import sys

def render_tikz_to_svg():
    """Render the TikZ diagram to SVG using pdflatex + pdf2svg"""
    
    # Create the LaTeX document
    tikz_content = r"""
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc, fit}

\begin{document}

% Define styles for the blocks
\tikzset{
    % Node styles
    input/.style={rectangle, rounded corners, draw=black, fill=blue!70!black, text=white, minimum height=1cm, minimum width=3cm, text width=3cm, align=center, thick},
    process/.style={rectangle, rounded corners, draw=black, fill=blue!50, text=white, minimum height=1cm, minimum width=3cm, text width=3cm, align=center, thick},
    block/.style={rectangle, rounded corners, draw=black, fill=cyan!60, text=white, minimum height=1cm, minimum width=3cm, text width=3cm, align=center, thick},
    add/.style={circle, draw=black, fill=gray!20, minimum size=0.8cm, thick},
    output/.style={rectangle, rounded corners, draw=black, fill=blue!70!black, text=white, minimum height=1cm, minimum width=3cm, text width=3cm, align=center, thick},
    
    % Box around the block
    fit_box/.style={draw, dashed, thick, inner sep=0.5cm, rounded corners, label={[anchor=east,xshift=-0.2cm]right:Transformer Block $n$}},
    
    % Arrow styles
    main_arrow/.style={-{Latex[length=3mm]}, thick},
    res_arrow/.style={-{Latex[length=3mm]}, thick, blue!60!black, dashed}
}

\begin{tikzpicture}[node distance=1cm and 1.5cm]

    % --- Input ---
    \node (input_tok) [input] {Input Tokens};
    \node (embed) [process, below=of input_tok] {Embedding Layer (Token + Position)};
    
    % --- Start of Blocks ---
    \node (dots_top) [below=of embed, yshift=0.2cm] {$\vdots$};
    \coordinate (block_in) at ($(dots_top.south) + (0, -0.5cm)$); % Input point for the block
    
    % --- Inside Transformer Block 'n' ---
    \node (ln1) [process, below=of block_in] {Layer Normalization};
    \node (mha) [block, below=of ln1] {Causal Multi-Head Attention};
    \node (add1) [add, below=of mha] {+};
    \node (ln2) [process, below=of add1] {Layer Normalization};
    \node (mlp) [block, below=of ln2] {MLP (Feed-Forward)};
    \node (add2) [add, below=of mlp] {+};

    % --- End of Blocks ---
    \coordinate (block_out) at ($(add2.south) + (0, -0.5cm)$); % Output point for the block
    \node (dots_bottom) [below=of block_out, yshift=0.2cm] {$\vdots$};
    
    % --- Output ---
    \node (final_norm) [process, below=of dots_bottom] {Final Layer Norm};
    \node (logits) [output, below=of final_norm] {Output Logits};
    \node (softmax) [output, below=of logits] {Softmax};
    \node (pred) [output, below=of softmax] {Next Token Prediction};
    
    % --- Draw Main Arrows ---
    \draw [main_arrow] (input_tok) -- (embed);
    \draw [main_arrow] (embed) -- (dots_top);
    \draw [main_arrow] (dots_top) -- (block_in);
    \draw [main_arrow] (block_in) -- (ln1);
    \draw [main_arrow] (ln1) -- (mha);
    \draw [main_arrow] (mha) -- (add1);
    \draw [main_arrow] (add1) -- (ln2);
    \draw [main_arrow] (ln2) -- (mlp);
    \draw [main_arrow] (mlp) -- (add2);
    \draw [main_arrow] (add2) -- (block_out);
    \draw [main_arrow] (block_out) -- (dots_bottom);
    \draw [main_arrow] (dots_bottom) -- (final_norm);
    \draw [main_arrow] (final_norm) -- (logits);
    \draw [main_arrow] (logits) -- (softmax);
    \draw [main_arrow] (softmax) -- (pred);
    
    % --- Draw Residual Arrows ---
    % Residual 1: Bypasses LN1 and MHA
    \draw [res_arrow] (block_in) -| ++(-2.5, 0) |- (add1.west);
    
    % Residual 2: Bypasses LN2 and MLP
    \draw [res_arrow] (add1.east) -| ++(2.5, 0) |- (add2.east);
    
    % --- Draw the 'fit' box ---
    \node [fit_box, fit=(ln1) (mha) (add1) (ln2) (mlp) (add2)] {};

\end{tikzpicture}

\end{document}
"""

    # Write to temporary file
    with open('transformer_architecture.tex', 'w') as f:
        f.write(tikz_content)
    
    try:
        # Compile to PDF
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'transformer_architecture.tex'], 
                      check=True, capture_output=True)
        
        # Convert PDF to SVG
        subprocess.run(['pdf2svg', 'transformer_architecture.pdf', 'transformer_architecture.svg'], 
                      check=True, capture_output=True)
        
        print("Successfully rendered TikZ diagram to transformer_architecture.svg")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error rendering TikZ: {e}")
        print("Make sure you have pdflatex and pdf2svg installed")
        return False
    except FileNotFoundError:
        print("Required tools not found. Please install:")
        print("  - pdflatex (part of TeX Live or MiKTeX)")
        print("  - pdf2svg")
        return False

if __name__ == "__main__":
    success = render_tikz_to_svg()
    if success:
        print("\nDiagram rendered! You can now use transformer_architecture.svg in your README")
    else:
        print("\nAlternative: Use online TikZ renderer like:")
        print("   - https://tikzcd.yichuanshen.de/")
        print("   - https://www.overleaf.com/")
        sys.exit(1)
