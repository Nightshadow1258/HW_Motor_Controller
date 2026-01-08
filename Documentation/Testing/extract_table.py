import pdfplumber
import pandas as pd

def extract_table_from_pdf(pdf_path, pages=None):
    """
    Extracts tables from specified pages of a PDF.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        pages (list or tuple): List of page numbers (1-based) or a tuple (start, end).
    
    Returns:
        pd.DataFrame: Combined table from selected pages.
    """
    all_rows = []
    header = None

    with pdfplumber.open(pdf_path) as pdf:
        if pages is None:
            selected_pages = pdf.pages
        elif isinstance(pages, tuple):
            selected_pages = pdf.pages[pages[0]-1:pages[1]]
        elif isinstance(pages, list):
            selected_pages = [pdf.pages[p-1] for p in pages]
        else:
            raise ValueError("Pages must be a list or tuple")

        for i, page in enumerate(selected_pages):
            tables = page.extract_tables()
            if not tables:
                continue

            table = tables[0]

            if header is None:
                header = table[0]
                rows = table[1:]
            else:
                if table[0] == header:
                    rows = table[1:]
                else:
                    rows = table

            all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=header)

# Example usage:
# pages = (2, 5)  # pages 2 to 5 inclusive
# pages = [2, 4, 6]  # specific pages
df = extract_table_from_pdf("stm32h750vb.pdf", pages=(58, 77))
df.to_csv("stm32h750vb_extracted_table.csv", index=False)
