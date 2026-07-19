import docx, re, os, shutil
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

SRC='/Users/kawa/work/kawa-technology/GitHub/metbit/manuscript/incob2026/Microsoft-Word-Proceeding-Templates/Bubpamala_metbit.docx'
OUTDIR='/Users/kawa/work/kawa-technology/GitHub/metbit/manuscript/CBC-journal/metbit-CBC'
os.makedirs(OUTDIR, exist_ok=True)
shutil.copy('/private/tmp/claude-501/-Users-kawa-work-kawa-technology-GitHub-metbit/9aa54781-b847-43a5-baca-54c587d23e34/scratchpad/media/image1.png', os.path.join(OUTDIR,'figure1.png'))
d=docx.Document(SRC)

EQS=[
 r"r_{j} = \mathrm{median}_{i}\, x_{ij},\qquad c_{i} = \mathrm{median}_{j}\!\left( \frac{x_{ij}}{r_{j}} \right),\qquad x_{ij}' = \frac{x_{ij}}{c_{i}}",
 r"x_{i} = a_{i} + b_{i}\,\bar{x} + \varepsilon_{i},\qquad x_{i}' = \frac{x_{i} - a_{i}}{b_{i}}",
 r"x_{ij}' = \frac{x_{ij}}{\sum_{k} x_{ik}}\,\kappa",
 r"\widetilde{x}_{ij} = \frac{x_{ij} - \bar{x}_{j}}{\sqrt{s_{j}}}",
 r"R^{2}X = 1 - \frac{\sum\left( X - \widehat{X} \right)^{2}}{\sum X^{2}},\qquad R^{2}Y = 1 - \frac{\sum\left( y - \widehat{y} \right)^{2}}{\sum y^{2}}",
 r"Q^{2} = 1 - \frac{\mathrm{PRESS}}{\mathrm{SS}_{Y}} = 1 - \frac{\sum_{i}\left( \widehat{y}_{i}^{\,cv} - y_{i} \right)^{2}}{\sum_{i} y_{i}^{2}}",
 r"r(a,j) = \frac{\mathrm{cov}\!\left( X_{a},X_{j} \right)}{\sigma_{a}\,\sigma_{j}},\qquad t_{j} = r(a,j)\sqrt{\frac{n_{j} - 2}{1 - r(a,j)^{2}}}",
 r"\mathrm{VIP}_{i} = \sqrt{\frac{p \cdot \sum_{h}\left( S_{h} \cdot \left( \frac{w_{ih}}{\lVert w_{\cdot h} \rVert} \right)^{2} \right)}{\sum_{h} S_{h}}}",
]

def esc(text):
    urls=[]; cites=[]
    def urlrepl(m):
        u=m.group(0); tail=''
        while u and u[-1] in '.,;:': tail=u[-1]+tail; u=u[:-1]
        urls.append(u); return f"\x00U{len(urls)-1}\x00"+tail
    text=re.sub(r'https?://[^\s)]+', urlrepl, text)
    def cr(m):
        nums=[x.strip() for x in m.group(1).split(',')]
        cites.append("\\cite{"+",".join("ref"+n for n in nums)+"}")
        return f"\x00C{len(cites)-1}\x00"
    text=re.sub(r'\[(\d+(?:,\s*\d+)*)\]', cr, text)
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('$',r'\$'),
                ('#',r'\#'),('_',r'\_'),('{',r'\{'),('}',r'\}'),
                ('~',r'\textasciitilde{}'),('^',r'\textasciicircum{}')]:
        text=text.replace(a,b)
    text=text.replace('x̄','$\\bar{x}$').replace('X̂','$\\hat{X}$')
    text=text.replace('̄','').replace('̂','')
    for a,b in {'×':'$\\times$','·':'$\\cdot$','–':'--','−':'$-$','ŷ':'$\\hat{y}$',
                'ê':'\\^{e}','ã':'\\~{a}','ü':'\\"{u}','κ':'$\\kappa$','±':'$\\pm$',
                ' ':' '}.items():
        text=text.replace(a,b)
    for i,u in enumerate(urls): text=text.replace(f"\x00U{i}\x00","\\url{"+u+"}")
    for i,c in enumerate(cites): text=text.replace(f"\x00C{i}\x00",c)
    # superscript runs (marked by rtext) -> \textsuperscript{}
    text=re.sub('\x02(.*?)\x03', lambda m: '\\textsuperscript{%s}'%m.group(1), text)
    return text

def rtext(p):
    """Paragraph text with superscript runs wrapped in markers, so esc() can
    emit \\textsuperscript{} (honours the docx superscript formatting for 1H, 10^-n, etc.)."""
    out=[]
    for r in p.runs:
        if not r.text: continue
        rpr=r._r.find(qn('w:rPr'))
        va=rpr.find(qn('w:vertAlign')) if rpr is not None else None
        sup = va is not None and va.get(qn('w:val'))=='superscript'
        out.append(('\x02'+r.text+'\x03') if sup else r.text)
    return ''.join(out)

# ---- first pass: frontmatter content ----
title=abstract=keywords=None
for p in d.paragraphs:
    s=p.style.name
    if s=='papertitle': title=rtext(p).strip()
    elif s=='abstract': abstract=rtext(p).strip()
    elif s=='keywords': keywords=rtext(p).strip()
abstract=re.sub(r'^Abstract\.\s*','',abstract)
kw=[k.strip() for k in re.sub(r'^Keywords:\s*','',keywords).split('·')]

# ---- body pass (document order) ----
BACK={'Author Contributions':'CRediT authorship contribution statement',
      'Acknowledgements':'Acknowledgements','Conflict of Interest':'Declaration of competing interest',
      'Funding':'Funding','Ethics Statement':'Ethics','Data Availability':'Data availability'}
refs=[]; out=[]; eq_i=0; pend_cap=None
FRONT={'papertitle','author','address','abstract','keywords','e-mail'}

def emit_table(tbl, caption):
    ncol=len(tbl.columns); spec='l'*ncol
    rows=[]
    for ri,row in enumerate(tbl.rows):
        cells=[esc(c.text.strip()) for c in row.cells]
        if ri==0: cells=[r'\textbf{%s}'%c for c in cells]
        rows.append(' & '.join(cells)+r' \\')
    tab=("\\begin{tabular}{%s}\n\\hline\n%s\n\\hline\n%s\n\\hline\n\\end{tabular}"
         %(spec, rows[0], '\n'.join(rows[1:])))
    return ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\caption{%s}\n"
            "\\resizebox{\\linewidth}{!}{%%\n%s}\n\\end{table}\n"%(caption, tab))

ARCH_FIG=r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[font=\footnotesize,>=latex,
   box/.style={draw,rounded corners,align=center,minimum height=8mm,inner sep=3pt}]
\node[box] (in) at (0,3.0) {Input DataFrame / NumPy array};
\node[box] (bi) at (0,1.7) {\texttt{backend\_info()} dispatch\\(matrix size, available hardware)};
\node[box] (gpu) at (-4.6,0) {GPU\\(experimental)};
\node[box] (c)   at (-1.55,0) {Native C\\(characterised)};
\node[box] (mp)  at (1.55,0) {Multiprocessing};
\node[box] (cn)  at (4.6,0) {Chunked NumPy\\(fallback)};
\draw[->] (in)--(bi);
\draw[->] (bi)--(gpu); \draw[->] (bi)--(c); \draw[->] (bi)--(mp); \draw[->] (bi)--(cn);
\end{tikzpicture}
\caption{Auto-dispatch backend architecture. A sample-by-feature matrix is routed by size and available hardware to one of four paths: an experimental GPU path (CuPy or PyTorch, not performance-characterised), the production-characterised native C kernels, a multiprocessing pool, or a memory-bounded chunked NumPy fallback. The selected backend is recorded in the fitted object.}
\label{fig:arch}
\end{figure}
"""

body=d.element.body
for child in body.iterchildren():
    if child.tag==qn('w:tbl'):
        out.append(emit_table(Table(child,d), pend_cap or '')); pend_cap=None; continue
    if child.tag!=qn('w:p'): continue
    p=Paragraph(child,d); s=p.style.name; txt=rtext(p).strip()
    if s in FRONT: continue
    if s=='equation':
        out.append("\\begin{equation}\n%s\n\\end{equation}\n"%EQS[eq_i]); eq_i+=1; continue
    if s=='image': continue  # handled at figurecaption
    if s=='figurecaption':
        cap=esc(re.sub(r'^Fig\.\s*\d+\.\s*','',txt))
        out.append("\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{figure1}\n"
                   "\\caption{%s}\n\\label{fig:workflow}\n\\end{figure}\n"%cap)
        out.append(ARCH_FIG); continue
    if s=='tablecaption':
        pend_cap=esc(re.sub(r'^Table\s*\d+\.\s*','',txt)); continue
    if s=='referenceitem':
        if txt: refs.append(txt); continue
    if s=='heading1':
        if txt.lower().startswith('references'): continue
        out.append("\n\\section{%s}\n"%esc(txt)); continue
    if s=='heading2':
        out.append("\n\\subsection{%s}\n"%esc(txt)); continue
    if s=='acknowlegments':
        lab,_,rest=txt.partition('. ')
        head=BACK.get(lab.strip(), lab.strip())
        out.append("\n\\section*{%s}\n%s\n"%(head, esc(rest.strip()))); continue
    if txt:
        out.append(esc(txt)+"\n")

body_tex='\n'.join(out)
bib='\n'.join("\\bibitem{ref%d}\n%s\n"%(i+1, esc(r)) for i,r in enumerate(refs))

HL=[ "Unified, scriptable workflow for preprocessing and chemometric analysis of NMR spectra",
     "Native kernels avoid full-matrix temporary copies in STOCSY and variance calculations",
     "VIP computation is about fivefold faster than vectorised NumPy at tested dimensions",
     "Native-kernel memory behaviour was reproduced across macOS and Linux",
     "Leakage-aware examples distinguish technical execution from biological validation"]

doc_tex=r"""%% !TEX root = main.tex
\documentclass[final,3p,times]{elsarticle}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\journal{Computational Biology and Chemistry}

\begin{document}

\begin{frontmatter}

\title{%s}

\author[kku,kawa]{Theerayut Bubpamala\corref{cor1}}
\ead{theerayut\_aeiw\_123@hotmail.com}
\author[kawa]{Chotika Chatgasem}

\affiliation[kku]{organization={Medical Biochemistry and Molecular Biology Graduate Study Program, Faculty of Medicine, Khon Kaen University},
            city={Khon Kaen},
            country={Thailand}}
\affiliation[kawa]{organization={kawa-technology, Independent Research \& Development},
            country={Thailand}}
\cortext[cor1]{Corresponding author.}

\begin{abstract}
%s
\end{abstract}

\begin{highlights}
%s
\end{highlights}

\begin{keyword}
%s
\end{keyword}

\end{frontmatter}

%s

\begin{thebibliography}{00}

%s
\end{thebibliography}

\end{document}
""" % (esc(title), abstract_tex if False else esc(abstract),
       '\n'.join(r'\item %s'%h for h in HL),
       ' \\sep '.join(esc(k) for k in kw),
       body_tex, bib)

open(os.path.join(OUTDIR,'main.tex'),'w').write(doc_tex)
print("wrote", os.path.join(OUTDIR,'main.tex'))
print("sections:", body_tex.count('\\section{'), "| subsections:", body_tex.count('\\subsection{'),
      "| equations:", body_tex.count('\\begin{equation}'), "| tables:", body_tex.count('\\begin{table}'),
      "| figures:", body_tex.count('\\begin{figure}'), "| refs:", len(refs))
