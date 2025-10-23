# How to Create the Transformer Architecture Diagram

Since GitHub doesn't natively render TikZ, here are the best options:

## Option 1: Online TikZ Renderer (Recommended)

1. **Go to**: https://tikzcd.yichuanshen.de/
2. **Copy the TikZ code** from the README
3. **Render to SVG/PNG**
4. **Download and save** as `transformer_architecture.png`
5. **Update README** to use the image

## Option 2: Local Rendering

If you have LaTeX installed:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install texlive-latex-extra pdf2svg

# Run the render script
python render_diagram.py
```

## Option 3: Overleaf

1. **Go to**: https://www.overleaf.com/
2. **Create new project**
3. **Paste the TikZ code**
4. **Compile and download** as PNG/SVG
5. **Add to repository**

## Quick Fix: Use Mermaid Instead

If you prefer to keep it in the README without external tools, we can convert back to Mermaid with the improved structure showing residual connections.

---

**Recommended**: Use Option 1 (online renderer) for the best quality and easiest workflow.
