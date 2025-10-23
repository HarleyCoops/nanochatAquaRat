# Alternative Mermaid Diagram for GitHub

Here's a Mermaid version that shows residual connections and renders directly on GitHub:

```mermaid
flowchart TD
    A[Input Tokens] --> B[Embedding Layer<br/>Token + Position]
    B --> C[Layer Normalization]
    C --> D[Causal Multi-Head<br/>Attention]
    D --> E[+]
    C -.->|Residual| E
    E --> F[Layer Normalization]
    F --> G[MLP<br/>Feed-Forward]
    G --> H[+]
    E -.->|Residual| H
    H --> I[Final Layer Norm]
    I --> J[Output Logits]
    J --> K[Softmax]
    K --> L[Next Token<br/>Prediction]
    
    style A fill:#1e3a8a,stroke:#1e40af,stroke-width:2px,color:#ffffff
    style B fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#ffffff
    style C fill:#0ea5e9,stroke:#0284c7,stroke-width:2px,color:#ffffff
    style D fill:#06b6d4,stroke:#0891b2,stroke-width:2px,color:#ffffff
    style E fill:#10b981,stroke:#059669,stroke-width:2px,color:#ffffff
    style F fill:#0ea5e9,stroke:#0284c7,stroke-width:2px,color:#ffffff
    style G fill:#06b6d4,stroke:#0891b2,stroke-width:2px,color:#ffffff
    style H fill:#10b981,stroke:#059669,stroke-width:2px,color:#ffffff
    style I fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#ffffff
    style J fill:#1e3a8a,stroke:#1e40af,stroke-width:2px,color:#ffffff
    style K fill:#1e3a8a,stroke:#1e40af,stroke-width:2px,color:#ffffff
    style L fill:#1e3a8a,stroke:#1e40af,stroke-width:2px,color:#ffffff
```

This version:
- ✅ Renders directly on GitHub
- ✅ Shows residual connections with dashed lines
- ✅ Uses the same color scheme
- ✅ Shows proper transformer architecture
- ✅ Includes softmax layer
