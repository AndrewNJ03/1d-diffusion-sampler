%%% CLASS DECLARATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ANStopical}[2023/08/08 ANFM 2025 Full Summary Template]

%%% Load BASE CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass[twoside,fleqn,letterpaper,11pt]{article}
\typeout{Loading ANStopical document class: includes [twoside,fleqn,letterpaper]{article}}

% 2024 - Original class file written for PHYSOR2024 to resemble the Word template.
% 2025 - Minor modifications to adapt template for ANFM2025

% \usepackage[bottom]{footmisc}  % Add the footmisc package for footnotes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Required Packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{titling}
\RequirePackage{titlesec}
\usepackage{setspace}
\RequirePackage{ragged2e}
\RequirePackage{enumitem}
\RequirePackage{amsmath}
\RequirePackage{graphicx}
\RequirePackage[letterpaper,margin=1in,headheight=0.5in, includefoot]{geometry}  % Set margins
\RequirePackage{newtxtext} % Turn on Times-New-Roman replaces times and mathpmtx
\RequirePackage{newtxmath} % math fonts for newtx
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[nameinlink,capitalize,noabbrev]{cleveref}
\RequirePackage[font=normalsize,
                labelfont=bf,
                textfont=bf,
                labelsep=period]{caption}
\RequirePackage{authblk}
\RequirePackage{apptools}
\RequirePackage{appendix}
\RequirePackage{ifpdf}
\ifpdf
  \RequirePackage{epstopdf} % pdflatex compatibility
  \pdfadjustspacing=1
\fi


\renewcommand{\footnoterule}{%
  \vspace*{34pt} % Adjust the vertical space as needed
  \hrule width 0.5\columnwidth%
  \vspace*{18pt} % Adjust the vertical space as needed
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Custom Commands and Class Errors for Users
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define addAuthor command and setup error handling
\newif\if@authorsdefined
\@authorsdefinedfalse
\newif\if@correspondingauthordefined
\@correspondingauthordefinedfalse
\newcommand\addAuthor[3][]{
    \@authorsdefinedtrue
    %\let\oldtextsuperscript\textsuperscript
    %\renewcommand{\textsuperscript}[1]{$^{\mathbf{#1}}$}
    \if\relax\detokenize{#1}\relax
        \author[#3]{\protect\noexpandarg{\protect\StrSubstitute{#2}{ }{\nobreakspace}}}
    \else
        \if@correspondingauthordefined
            \ClassError{ANStopical}{Only one corresponding author is allowed}{remove [] from addAuthor}
        \else
            \author[#3,\protect\footnotemark]{\protect\noexpandarg{\protect\StrSubstitute{#2}{ }{\nobreakspace}} \protect\blfootnote{ #1}}
        \fi
        \@correspondingauthordefinedtrue
    \fi
    %\renewcommand{\textsuperscript}[1]{\oldtextsuperscript{#1}}
}

% Define addAffiliation command and setup error handling
\newif\if@affilsdefined
\@affilsdefinedfalse
\newcommand{\addAffiliation}[2]{
    \affil[#1]{\protect\StrSubstitute{#2}{ }{\nobreakspace}}
    \@affilsdefinedtrue
}

%Define Abstract command and error handling
\newif\if@abstractdefined
\@abstractdefinedfalse
\def\@abstract{\@empty}
\newcommand{\Abstract}[1]{\def\@abstract{#1}\@abstractdefinedtrue}


%Define keywords command and error handling
\newif\if@keywordsdefined
\@keywordsdefinedfalse
\def\@keywords{\@empty}
\newcommand{\keywords}[1]{\def\@keywords{#1}\@keywordsdefinedtrue}

%Define shortTitle command and error handling
\newif\if@shortTitledefined
\@shortTitledefinedfalse
\def\@shortTitle{\@empty}
\newcommand{\shortTitle}[1]{\def\@shortTitle{#1}\@shortTitledefinedtrue}


%Define Author Header command and error handling
\newif\if@authorHeaddefined
\@authorHeaddefinedfalse
\def\@authorHead{\@empty}
\newcommand{\authorHead}[1]{\def\@authorHead{#1}\@authorHeaddefinedtrue}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Font size definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand\Large{\@setfontsize\Large{14pt}{16.1pt}} % For title
\renewcommand\large{\@setfontsize\large{12pt}{13.8pt}} % For section headings
\renewcommand\normalsize{\@setfontsize\normalsize{11pt}{12.65pt}}
\renewcommand\small{\@setfontsize\small{10pt}{11.5pt}}   % For references and header/footer


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formatting of Preamble (Title, authors, institution, \maketitle)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand\Authfont{\bfseries\normalsize}  %Set font styles for authors
\renewcommand\Affilfont{\normalfont}          %Set font styles for affiliations
\renewcommand\Authand{, }  %remove word "and" when 2 authors
\renewcommand\Authands{, } %remove word "and" when 3 or more authors
\renewcommand\AB@affilsepx{; \protect\Affilfont} % Set affiliation separator to ;
\renewcommand\AB@authnote[1]{\textsuperscript{\normalfont\bfseries#1}}

% Change how title, authors, and affiliations are displayed
\renewcommand{\@maketitle}
{
    \newpage
    \null    
    \let\footnote\thanks
    % Define custom margins for the first page only
    \begin{center}%
        %\vspace{-0.75in}
        \vspace{-20pt}
        {\normalfont\Large\bfseries\begin{tabular}[t]{c}\@title\end{tabular}}%
        
        \vspace{7pt} % Space after title
        \if@authorsdefined
            \if@affilsdefined
                {\begin{tabular}[t]{c}\@author\end{tabular}\par}%
                %\vspace{6pt} % Space after authors
            \else
                \ClassError{ANStopical}{No author affiliations have been defined}{Please use the addAffiliation command before begin{document}}
            \fi
        \else
            \ClassError{ANStopical}{No authors have been defined}{Please use the addAuthor command before begin{document}}
        \fi
        \vspace{6pt} % Space after authors
        \emph{[leave space for DOI, which will be inserted by ANS]}
        \vspace{2pt} % Space after authors
    \end{center}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers and such (fancyhdr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtBeginDocument{

    % Set PDF Metadata (this is really hard to do correctly)
    % \hypersetup{
    %     pdfauthor={\authors},
    %     pdftitle={\@title},
    %     pdfsubject={ANFM 2025},
    %     pdfkeywords={\@keywords},
    % }
% Define custom margins for the first page only
    \maketitle
    \makeAbstract
    \makeKeywords

    % Clear footer and header
    \fancyhf{}
    \thispagestyle{empty}
    \pagestyle{empty}
    

    %Define Header and Footer on First Page
    % \fancypagestyle{firstpage}{
    %     \fancyhead[C]{
    %         \small{\emph{
    %             ANFM 2025 - Advances in Nuclear Fuel Management $\cdot$ Clearwater, Florida, USA $\cdot$ July 20 -- 23, 2025
    %         }}
    %     }
    %     \lfoot{}
    % }
    % \thispagestyle{firstpage}

    % All other pages
    % \pagestyle{fancy}
    % \if@authorHeaddefined 
    %     \fancyhead[CE]{{\small \@authorHead}}
    % \else
    %     \ClassError{ANStopical}{No short author header has been defined}{Please use the authHead command before begin{document}}
    % \fi
    
    % \if@shortTitledefined
    %     \fancyhead[CO]{{\small \@shortTitle}}
    % \else
    %     \ClassError{ANStopical}{No short title has been defined}{Please use the shortTitle command before begin{document}}
    % \fi
    
    % \rfoot{\small \thepage/\pageref*{LastPage}}
    % \lfoot{
    %     \footnotesize{\emph{
    %             ANFM 2025 - Advances in Nuclear Fuel Management $\cdot$ Clearwater, Florida, USA $\cdot$ July 20 -- 23, 2025
    %         }}
    % }

    %%% Set miscellaneous page layout stuff
    \justify
    \renewcommand{\headrulewidth}{0pt}  % No bars at header/footer
    \urlstyle{rm}
    \setlength{\parskip}{6pt}   %Set space between paragraphs
    % \setlength{\baselineskip}{12pt} % Set the baseline skip to achieve single spacing
    \setlength\parindent{0pt} %Remove indentation, must be called after \justify
    \setlist{nolistsep}
    \renewcommand{\thetable}{\Roman{table}}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section title definitions and formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \renewcommand{\section}{
%   \renewcommand{\@seccntformat}[1]{\csname the##1\endcsname.\hspace{0.5em}}
%   \@startsection{section}{1}{\z@}
%                 {1.0ex plus 1.0ex minus 0.2ex}{0.2ex plus 0.2ex}{\normalsize\bf\center}}
% \renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}
%   {1.0ex plus 1.0ex minus 0.2ex}{0.2ex plus 0.2ex}{\normalsize\bf}}
% \renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}
%   {1.0ex plus 1.0ex minus 0.2ex}{0.2ex plus 0.2ex}{\normalsize\bf}}
% %
% \renewcommand{\thesection}{\@arabic\c@section}
% \renewcommand{\thesubsection}{\thesection.\@arabic\c@subsection}
% \renewcommand{\thesubsubsection}{\thesubsection.\@arabic\c@subsubsection}

\titleformat{\section}{\bf\large\centering}{\IfAppendix{\MakeUppercase\appendixname \,}{}\thetitle.}{4ex}{\MakeUppercase}[]
\titleformat{\subsection}{\bf\large}{\thetitle.}{3ex}{}[]
\titleformat{\subsubsection}{\bf\large}{\thetitle.}{2ex}{}[]
\titlespacing{\section}{0pt}{8pt plus 0pt minus 0pt}{6pt plus 0pt minus 0pt}
\titlespacing{\subsection}{0pt}{6pt plus 0pt minus 0pt}{6pt plus 0pt minus 0pt}
\titlespacing{\subsubsection}{0pt}{6pt plus 0pt minus 0pt}{6pt plus 0pt minus 0pt}

\renewcommand{\refname}{REFERENCES}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Miscellaneous Custom Commands used in this Class file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\blfootnote[1]{%
  \begingroup
  \fontsize{10pt}{14pt}\selectfont % Font size 10pt, line spacing exactly 14pt
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}

\newcommand{\makeAbstract}{
    \if@abstractdefined
        \renewenvironment{abstract}{
            \centering
            \setlength{\parskip}{18pt} % Space before the abstract
            \large\textbf{ABSTRACT}
            \vspace{0.5em}
            \begin{list}{}
                {\setlength{\leftmargin}{0.5in}
                \setlength{\rightmargin}{0.5in}
                \setlength{\topsep}{0in}\small}
                
                \item
                
        }{\end{list}}
        
        \begin{abstract}
            \justify
            \noindent
            \@abstract
        \end{abstract}
    \else
        \ClassError{ANStopical}{No abstract has been defined}{Please use the abstract command before begin{document}}
    \fi
}

\newcommand{\makeKeywords}{
    \if@keywordsdefined
        \begin{raggedright}
            \begin{list}{}
            {\setlength{\leftmargin}{0.5in}
             \setlength{\rightmargin}{0.5in}\small}
             
                \item[] \ignorespaces \emph{Keywords:} \upshape \@keywords
                
            \end{list}
        \end{raggedright}
    \else
        \ClassError{ANStopical}{No keywords have been defined}{Please use the keywords command before begin{document}}
    \fi
}
