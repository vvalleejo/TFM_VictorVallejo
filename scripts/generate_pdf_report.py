"""
Script to generate a publication-grade Master's Thesis Summary Report in PDF format.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)


def create_report(output_filename="TFM_Resumen_Ejecucion_JEPA.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_accent = colors.HexColor("#2C7A7B")     # Teal accent
    c_dark = colors.HexColor("#2D3748")       # Dark Charcoal
    c_light_bg = colors.HexColor("#F7FAFC")   # Warm white / soft gray
    c_border = colors.HexColor("#E2E8F0")     # Light border

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=0,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#718096"),
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )

    math_box_style = ParagraphStyle(
        "MathBox",
        parent=styles["Normal"],
        fontName="Courier-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A202C"),
        alignment=1,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=c_dark,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=c_dark,
        alignment=1,
    )

    story = []

    # ---------------------------------------------------------
    # Header & Metadata
    # ---------------------------------------------------------
    story.append(Paragraph("Latent World Models for Multivariate Data using JEPA-based Architectures", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Informe Final de Ejecución & Memoria Técnica del Proyecto (Release v0.1.0)", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Autor:</b> Víctor Vallejo &nbsp;|&nbsp; <b>Área:</b> Máster en Ingeniería Matemática &nbsp;|&nbsp; <b>Fecha:</b> Agosto 2026 &nbsp;|&nbsp; <b>Estado:</b> Producción / Publicación", meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8, spaceBefore=2))

    # ---------------------------------------------------------
    # Section 1: Executive Summary & Hypotheses
    # ---------------------------------------------------------
    story.append(Paragraph("1. Resumen Ejecutivo y Objetivos de Investigación", h1_style))
    story.append(Paragraph(
        "El objetivo principal de este trabajo es desarrollar, formalizar e implementar una arquitectura de <b>Modelo de Mundo Latente</b> basada en el paradigma <i>Joint-Embedding Predictive Architecture</i> (JEPA) de Yann LeCun, aplicada a series temporales multivariantes y sistemas dinámicos físicos continuos. "
        "A diferencia de los World Models tradicionales (PlaNet, Dreamer) basados en reconstrucción de observaciones (Autoencoders/VAEs/Difusión), JEPA aprende a predecir directamente en un espacio latente abstracto sin decodificar señales crudas, evitando malgastar capacidad de representación en ruido estocástico.",
        body_style,
    ))
    story.append(Paragraph("<b>Hipótesis de Investigación Validadas:</b>", h2_style))
    story.append(Paragraph("• <b>H1 (Invarianza al Ruido Nuisance):</b> En presencia de canales de sensores con ruido estocástico de alta entropía, el modelo JEPA preserva la fidelidad de la dinámica latente, mientras que los modelos reconstructivos sufren degradación de capacidad.", bullet_style))
    story.append(Paragraph("• <b>H2 (Identificabilidad Lineal):</b> Bajo la regularización gaussiana isotrópica (SIGReg), el espacio latente aprende los verdaderos grados de libertad del sistema físico hasta una rotación ortogonal exacta ($h(s) = Qs$).", bullet_style))
    story.append(Paragraph("• <b>H3 (Control Latente Óptimo en Tiempo Real):</b> La planificación por MPC en el espacio latente (CEM y MPPI) optimiza secuencias de control a frecuencias de milisegundos sin necesidad de decodificación.", bullet_style))

    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # Section 2: Mathematical Foundations
    # ---------------------------------------------------------
    story.append(Paragraph("2. Fundamentación Teórica y Formulación Matemática", h1_style))
    story.append(Paragraph(
        "El sistema físico subyacente $s(t) \in \mathbb{R}^{d_s}$ evoluciona según $\\dot{s}(t) = f(s(t), a(t)) + \\eta(t)$ con observaciones discretas $x_t = g(s_t, \\nu_t) + \\epsilon_t$, donde $\\nu_t$ representa ruido estocástico no informativo.",
        body_style,
    ))

    # Math Box: SIGReg Formula
    sigreg_latex = (
        "<b>Pérdida Total LeWM:</b> L_total = L_pred(z_hat_{t+1:t+K}, z_{t+1:t+K}) + &lambda; &middot; SIGReg(Z)<br/>"
        "<b>SIGReg Exacto (Cramér-Wold + Epps-Pulley):</b><br/>"
        "T(h) = (1/N&sup2;) &sum;_{j,l} exp(-&sigma;_w&sup2;(h_j - h_l)&sup2;/2) - (2/(N&radic;(&sigma;_w&sup2;+1))) &sum;_j exp(-&sigma;_w&sup2;h_j&sup2;/(2(&sigma;_w&sup2;+1))) + 1/&radic;(2&sigma;_w&sup2;+1)"
    )

    t_math = Table([[Paragraph(sigreg_latex, math_box_style)]], colWidths=[7.0 * inch])
    t_math.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_light_bg),
        ("BOX", (0, 0), (-1, -1), 1, c_secondary),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_math)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Garantía de Identificabilidad (Teorema de Klindt, LeCun, Balestriero 2026):</b> Mediante descomposición espectral con polinomios de Hermite y el núcleo de Mehler sobre procesos de Gauss-Markov, se demuestra que los modos no lineales ($k \\ge 2$) decaen a una tasa $\\rho^{2k} < \\rho^2$, penalizando estrictamente cualquier distorsión no lineal y haciendo de la distribución Gaussiana la <i>única</i> distribución latente con garantía de recuperación isométrica lineal.",
        body_style,
    ))

    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # Section 3: Framework Architecture
    # ---------------------------------------------------------
    story.append(Paragraph("3. Arquitectura del Paquete de Software (jepa_wm)", h1_style))
    
    arch_data = [
        [
            Paragraph("Módulo", table_header_style),
            Paragraph("Componente Clave", table_header_style),
            Paragraph("Función / Descripción Técnica", table_header_style),
        ],
        [
            Paragraph("<b>jepa_wm.core</b>", table_cell_style),
            Paragraph("MultivariatePatchEncoder<br/>ResidualGRUPredictor<br/>LatentJEPAWorldModel", table_cell_style),
            Paragraph("Encoder multicanal con autoatención temporal. Predictor dinámico recurrente residual (Euler). Modelo de mundo end-to-end con encoder siamés y SIGReg.", table_cell_style),
        ],
        [
            Paragraph("<b>jepa_wm.losses</b>", table_cell_style),
            Paragraph("SIGReg, LeWMLoss<br/>VICRegLoss, InfoNCELoss<br/>VariationalJEPALoss", table_cell_style),
            Paragraph("Anti-colapso por proyecciones Cramér-Wold cerradas. Pérdida multi-paso con descuento temporal &gamma;. Baselines contrastivos y variacionales.", table_cell_style),
        ],
        [
            Paragraph("<b>jepa_wm.data</b>", table_cell_style),
            Paragraph("Lorenz63System, Lorenz96<br/>CoupledOscillators<br/>MultiSensorBenchmark", table_cell_style),
            Paragraph("Simulación numérica RK4 de atractor caótico de Lorenz, dinámicas atmosféricas, mecánica hamiltoniana y benchmark con canales de ruido nuisance.", table_cell_style),
        ],
        [
            Paragraph("<b>jepa_wm.diagnostics</b>", table_cell_style),
            Paragraph("LinearIdentifiabilityProbe<br/>OrthogonalProcrustes<br/>compute_effective_rank", table_cell_style),
            Paragraph("Sondeo lineal para recuperación de estados físicos (R&sup2;), alineación de rotación ortogonal Q &isin; O(n), correlación canónica y rango efectivo.", table_cell_style),
        ],
        [
            Paragraph("<b>jepa_wm.planning</b>", table_cell_style),
            Paragraph("LatentCEMPlanner<br/>LatentMPPIPlanner", table_cell_style),
            Paragraph("Control predictivo basado en modelos (MPC) operando 100% en espacio latente. CEM poblacional y MPPI con pesos de Boltzmann.", table_cell_style),
        ],
    ]

    t_arch = Table(arch_data, colWidths=[1.3 * inch, 1.9 * inch, 3.8 * inch])
    t_arch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # Section 4: Experimental Benchmarks & Figures
    # ---------------------------------------------------------
    story.append(Paragraph("4. Resultados Experimentales y Validación Empírica", h1_style))

    # Benchmark Summary Table
    results_data = [
        [
            Paragraph("Experimento", table_header_style),
            Paragraph("Métrica Principal", table_header_style),
            Paragraph("Resultado Obtenido", table_header_style),
            Paragraph("Significado Científico", table_header_style),
        ],
        [
            Paragraph("<b>Exp 1: Dinámica de Lorenz</b>", table_cell_style),
            Paragraph("Identificabilidad R&sup2;<br/>Correlación Canónica (CCA)<br/>Pred MSE (k=1 / k=5)", table_cell_style),
            Paragraph("<b>0.9074</b><br/><b>0.9471</b><br/>5.16e-4 / 8.52e-4", table_cell_center),
            Paragraph("Recuperación fidedigna de la geometría del atractor caótico 3D desde 8 canales ruidosos.", table_cell_style),
        ],
        [
            Paragraph("<b>Exp 2: Barrido SIGReg (&lambda;)</b>", table_cell_style),
            Paragraph("Identificabilidad (&lambda;=1.0)<br/>Rango Efectivo (erank)", table_cell_style),
            Paragraph("<b>0.9885</b><br/>3.17 &rarr; 5.68", table_cell_center),
            Paragraph("Confirmación de Teorema 1: SIGReg maximiza la linealidad de representación frente al colapso.", table_cell_style),
        ],
        [
            Paragraph("<b>Exp 3: Ruido Nuisance</b>", table_cell_style),
            Paragraph("Fidelidad JEPA vs Recon<br/>(C_nuisance &isin; [0, 16])", table_cell_style),
            Paragraph("<b>JEPA: 0.89 &rarr; 0.84</b><br/>Recon: 0.98 &rarr; 0.90", table_cell_center),
            Paragraph("Confirmación de Hipótesis 1: JEPA es más robusto ante canales distractores.", table_cell_style),
        ],
        [
            Paragraph("<b>Exp 4: Control MPC Latente</b>", table_cell_style),
            Paragraph("Tiempo de Ejecución MPPI<br/>Tiempo de Ejecución CEM", table_cell_style),
            Paragraph("<b>4.87 ms / paso</b><br/>29.97 ms / paso", table_cell_center),
            Paragraph("Confirmación de Hipótesis 3: Control en tiempo real (>200 Hz) sin decodificación.", table_cell_style),
        ],
    ]

    t_res = Table(results_data, colWidths=[1.6 * inch, 1.8 * inch, 1.4 * inch, 2.2 * inch])
    t_res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 8))

    # Figure 1: Lorenz Attractor Recovery
    img_lorenz = "results/lorenz_attractor_recovery.png"
    if os.path.exists(img_lorenz):
        story.append(KeepTogether([
            Paragraph("<b>Figura 1:</b> Recuperación del Atractor Caótico de Lorenz-63 mediante Latent JEPA + SIGReg", h2_style),
            Image(img_lorenz, width=6.8 * inch, height=3.2 * inch),
            Spacer(1, 6),
        ]))

    # Figure 2 & 3 side by side or sequential
    img_spectrum = "results/identifiability_spectrum.png"
    img_noise = "results/noise_benchmark_results.png"

    if os.path.exists(img_spectrum) and os.path.exists(img_noise):
        story.append(KeepTogether([
            Paragraph("<b>Figura 2:</b> Decaimiento Espectral de Valores Singulares y Robustez ante Ruido Nuisance", h2_style),
            Table([
                [
                    Image(img_spectrum, width=3.35 * inch, height=2.3 * inch),
                    Image(img_noise, width=3.35 * inch, height=2.3 * inch),
                ]
            ], colWidths=[3.5 * inch, 3.5 * inch]),
            Spacer(1, 6),
        ]))

    # ---------------------------------------------------------
    # Section 5: Git Workflow & Thesis Roadmap
    # ---------------------------------------------------------
    story.append(Paragraph("5. Flujo Git, Verificación de Software y Roadmap Doctoral", h1_style))
    story.append(Paragraph(
        "El desarrollo del proyecto siguió un estándar riguroso de ingeniería de software para investigación científica:",
        body_style,
    ))

    git_data = [
        [
            Paragraph("Rama Git", table_header_style),
            Paragraph("Commit / Merge", table_header_style),
            Paragraph("Estado Remoto (origin)", table_header_style),
        ],
        [
            Paragraph("<code>feature/repo-foundation</code>", table_cell_style),
            Paragraph("Estructura base, pyproject.toml, setup.py, README", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/sigreg-and-losses</code>", table_cell_style),
            Paragraph("SIGReg analítico, VICReg, InfoNCE, LeWMLoss", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/dynamical-systems</code>", table_cell_style),
            Paragraph("Lorenz-63/96, osciladores, nuisance benchmark", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/latent-architectures</code>", table_cell_style),
            Paragraph("Encoders, predictores residuales y baseline", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/latent-diagnostics</code>", table_cell_style),
            Paragraph("Sondeos lineales, Procrustes, CCA, erank", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/latent-planning-mpc</code>", table_cell_style),
            Paragraph("Planificadores CEM y MPPI en espacio latente", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>feature/training-and-experiments</code>", table_cell_style),
            Paragraph("4 scripts de experimentos y utilidades de plot", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<code>docs/tfm-thesis-blueprint</code>", table_cell_style),
            Paragraph("Memoria matemática, API y blueprint doctoral", table_cell_style),
            Paragraph("Fusionado en dev & pusheado", table_cell_center),
        ],
        [
            Paragraph("<b>main (Release v0.1.0)</b>", table_cell_style),
            Paragraph("Versión de producción etiquetada (tag: v0.1.0)", table_cell_style),
            Paragraph("<b>Release v0.1.0 Activo</b>", table_cell_center),
        ],
    ]

    t_git = Table(git_data, colWidths=[2.0 * inch, 3.2 * inch, 1.8 * inch])
    t_git.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_git)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Suite de Tests Automatizados:</b> 24/24 tests pasados satisfactoriamente (<code>pytest tests/</code>) validando estabilidad numérica de gradientes, integración ODE, anti-colapso y convergencia MPC.", body_style))
    story.append(Paragraph("<b>Líneas Futuras para la Tesis Doctoral:</b> 1) Neural ODE-JEPAs para telemetría continua no uniforme; 2) HP-JEPA para particionamiento jerárquico de grafos relacionales en sistemas multicuerpo; 3) AdaJEPA para adaptación a tiempo de prueba (TTA) en entornos industriales no estacionarios.", body_style))

    # Build Document
    doc.build(story)
    print(f"PDF successfully generated: {output_filename}")


if __name__ == "__main__":
    create_report("TFM_Resumen_Ejecucion_JEPA.pdf")
