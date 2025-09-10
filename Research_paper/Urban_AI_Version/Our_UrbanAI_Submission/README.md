# Urban AI Conference Paper Adaptation

This folder contains the paper adapted from the NeurIPS 2025 format to the ACM Urban AI conference format.

## Files

- `urban_ai_paper.tex` - Main paper file in ACM format
- `references.bib` - Bibliography file with all references
- `README.md` - This explanation file

## Key Changes Made

### Format Conversion
- **Document class**: Changed from `\documentclass{article}` with `neurips_2025` to `\documentclass[sigconf,review]{acmart}`
- **Template**: Switched from NeurIPS template to ACM sigconf template for conference submissions
- **Bibliography**: Converted from inline `thebibliography` to external `.bib` file with `ACM-Reference-Format` style

### ACM-Specific Requirements Added
- **Copyright information**: Added ACM copyright settings
- **Conference details**: Added conference information placeholders for Urban AI conference
- **CCS concepts**: Added Computing Classification System concepts relevant to the paper
- **Keywords**: Formatted keywords according to ACM requirements
- **Author format**: Converted to ACM author/affiliation format with proper email and ORCID support

### Content Adaptations
- **Abstract**: Refined to be more concise and impactful for Urban AI audience
- **Introduction**: Enhanced urban context and smart city applications
- **Sections**: Maintained technical content but adjusted presentation for urban AI focus
- **Results**: Emphasized urban planning and city management applications
- **Discussion**: Added broader urban analytics implications and smart city integration points

### Urban AI Focus Enhancements
- Emphasized smart city applications and urban planning insights
- Highlighted interpretability for city planners and transportation operators  
- Added discussion of equity and societal impact considerations
- Expanded on operational applications for city management
- Included cross-city generalization results relevant to urban AI deployment

## Compilation Instructions

To compile the paper:

1. Ensure you have the ACM template files in the same directory or accessible path
2. Use pdflatex or your preferred LaTeX compiler:
   ```bash
   pdflatex urban_ai_paper.tex
   bibtex urban_ai_paper
   pdflatex urban_ai_paper.tex
   pdflatex urban_ai_paper.tex
   ```

## ACM Template Requirements

The paper uses the `acmart` document class which should be available with modern LaTeX distributions. If you encounter issues, ensure you have:
- `acmart.cls` - The ACM article class
- `ACM-Reference-Format.bst` - Bibliography style file

These are typically included with the ACM template package available from the ACM website.

## Review vs. Final Version

Currently set up for review submission with `[sigconf,review]` options. For camera-ready submission, change to:
```latex
\documentclass[sigconf]{acmart}
```

## Conference Information

Conference details (dates, location, etc.) need to be updated when actual Urban AI conference information is available. Current placeholders should be replaced with:
- Actual conference name and dates
- Conference location  
- ISBN and DOI information
- Submission ID if provided

## Technical Content

The core technical contributions remain unchanged:
- Multi-scale OpenStreetMap feature integration
- Attention-based spatial scale fusion  
- Enhanced DCRNN architecture
- Experimental validation on Swiss bike-sharing data
- Urban planning insights and interpretable results

The adaptation focuses on presentation and format while preserving the scientific rigor and technical innovations of the original work.
