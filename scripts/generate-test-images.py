#!/usr/bin/env python3
"""Generate diverse OCR benchmark test images with known ground truth."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import textwrap

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "images")
GT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "ground-truth")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

def get_font(size=16):
    """Get a basic font."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def get_serif_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return get_font(size)

def get_bold_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return get_font(size)

def save(img, name, gt_text):
    img.save(os.path.join(OUT_DIR, f"{name}.png"))
    with open(os.path.join(GT_DIR, f"{name}.txt"), "w") as f:
        f.write(gt_text.strip() + "\n")
    print(f"  ✓ {name}.png + ground truth")


# === 1. Dense paragraph text ===
def gen_dense_paragraph():
    text = (
        "The development of optical character recognition technology has transformed how we interact with "
        "printed and handwritten text. Modern OCR systems leverage deep learning architectures, particularly "
        "convolutional neural networks and transformer models, to achieve unprecedented accuracy rates. These "
        "systems can process documents in hundreds of languages, handle varying font styles, and even interpret "
        "degraded or noisy input images. The field continues to evolve rapidly, with recent advances in "
        "vision-language models pushing the boundaries of what automated text recognition can accomplish. "
        "Enterprise applications range from invoice processing and contract analysis to medical record "
        "digitization and historical document preservation. The accuracy of modern OCR engines typically "
        "exceeds 95% for clean printed text, though performance varies significantly when dealing with "
        "handwritten content, complex layouts, or low-resolution scans. Benchmarking these systems requires "
        "careful methodology, including diverse test corpora and standardized evaluation metrics."
    )
    img = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(img)
    font = get_serif_font(14)
    wrapped = textwrap.fill(text, width=85)
    draw.text((40, 40), wrapped, fill="black", font=font)
    save(img, "dense_paragraph", text)

# === 2. Multi-column layout (newspaper style) ===
def gen_multi_column():
    col1 = (
        "TECHNOLOGY NEWS\n\n"
        "OpenOCR Launches New\n"
        "Benchmark Framework\n\n"
        "The open-source OCR platform\n"
        "announced today its new\n"
        "benchmarking suite, designed\n"
        "to standardize quality\n"
        "measurement across engines.\n"
        "The framework includes six\n"
        "metrics covering accuracy,\n"
        "gibberish detection, and\n"
        "ground-truth comparison."
    )
    col2 = (
        "MARKET UPDATE\n\n"
        "OCR Market Expected to\n"
        "Reach $32B by 2030\n\n"
        "Industry analysts project\n"
        "continued strong growth in\n"
        "the document processing\n"
        "sector. Key drivers include\n"
        "regulatory compliance needs\n"
        "and the push toward digital\n"
        "transformation across all\n"
        "industries worldwide."
    )
    gt = col1.replace("\n\n", " ").replace("\n", " ") + " " + col2.replace("\n\n", " ").replace("\n", " ")
    img = Image.new("RGB", (800, 450), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(13)
    bold = get_bold_font(15)
    # Draw dividing line
    draw.line([(395, 30), (395, 420)], fill="gray", width=1)
    # Col 1
    y = 40
    for line in col1.split("\n"):
        f = bold if line.isupper() or "OpenOCR" in line or "Benchmark" in line else font
        draw.text((40, y), line, fill="black", font=f)
        y += 20 if line else 10
    # Col 2
    y = 40
    for line in col2.split("\n"):
        f = bold if line.isupper() or "OCR Market" in line or "Reach" in line else font
        draw.text((420, y), line, fill="black", font=f)
        y += 20 if line else 10
    save(img, "multi_column", gt)

# === 3. Table/form ===
def gen_table_form():
    headers = ["Engine", "Accuracy", "Latency", "License"]
    rows = [
        ["Tesseract", "97.8%", "350ms", "Apache-2.0"],
        ["Surya", "98.0%", "2800ms", "GPL-3.0"],
        ["EasyOCR", "93.7%", "2500ms", "Apache-2.0"],
        ["docTR", "92.0%", "800ms", "Apache-2.0"],
        ["OCRmyPDF", "97.6%", "500ms", "MPL-2.0"],
        ["RapidOCR", "82.9%", "200ms", "Apache-2.0"],
    ]
    gt_lines = ["OCR Engine Comparison Report", ""]
    gt_lines.append(" | ".join(headers))
    for row in rows:
        gt_lines.append(" | ".join(row))

    img = Image.new("RGB", (700, 350), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(13)
    bold = get_bold_font(14)
    draw.text((40, 30), "OCR Engine Comparison Report", fill="black", font=get_bold_font(18))
    # Table
    col_widths = [150, 100, 100, 120]
    x_start = 40
    y_start = 80
    # Headers
    x = x_start
    for i, h in enumerate(headers):
        draw.text((x + 5, y_start + 5), h, fill="black", font=bold)
        x += col_widths[i]
    draw.line([(x_start, y_start + 25), (x_start + sum(col_widths), y_start + 25)], fill="black", width=2)
    # Rows
    for ri, row in enumerate(rows):
        y = y_start + 30 + ri * 30
        x = x_start
        for ci, cell in enumerate(row):
            draw.text((x + 5, y + 5), cell, fill="black", font=font)
            x += col_widths[ci]
        if ri < len(rows) - 1:
            draw.line([(x_start, y + 28), (x_start + sum(col_widths), y + 28)], fill="gray", width=1)
    # Border
    draw.rectangle(
        [(x_start, y_start), (x_start + sum(col_widths), y_start + 30 + len(rows) * 30)],
        outline="black", width=2
    )
    save(img, "table_form", "\n".join(gt_lines))

# === 4. Low quality / noisy scan ===
def gen_noisy_scan():
    text = (
        "NOTICE OF MEETING\n\n"
        "Date: March 15, 2026\n"
        "Time: 2:00 PM EST\n"
        "Location: Conference Room B\n\n"
        "All department heads are required to attend the quarterly review meeting. "
        "Please bring your Q1 performance reports and budget proposals for Q2. "
        "Remote attendance is available via the usual video conference link."
    )
    img = Image.new("RGB", (600, 400), "#f0ead6")  # aged paper color
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    bold = get_bold_font(16)
    y = 40
    for line in text.split("\n"):
        if not line:
            y += 10
            continue
        f = bold if line.isupper() or line.startswith("Date") or line.startswith("Time") or line.startswith("Location") else font
        # Slight random offset for "scanned" look
        x_off = random.randint(-1, 1)
        draw.text((40 + x_off, y), line if len(line) < 80 else textwrap.fill(line, 55), fill="#1a1a1a", font=f)
        y += 20 if len(line) < 60 else 50
    # Add noise using Pillow only
    import struct
    pixels = img.load()
    w, h = img.size
    for py in range(h):
        for px in range(w):
            r, g, b = pixels[px, py]
            n = random.randint(-20, 20)
            pixels[px, py] = (max(0, min(255, r+n)), max(0, min(255, g+n)), max(0, min(255, b+n)))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    # Slight rotation for scan effect
    img = img.rotate(1.5, fillcolor="#f0ead6", expand=False)
    gt = text.replace("\n\n", "\n").replace("\n", "\n")
    save(img, "noisy_meeting_notice", text)

# === 5. Mixed fonts / styles ===
def gen_mixed_fonts():
    title = "Product Specification Sheet"
    sections = [
        ("Model:", "OCR-X500 Professional"),
        ("Manufacturer:", "OpenOCR Technologies Inc."),
        ("Weight:", "2.4 kg"),
        ("Dimensions:", "320 x 240 x 45 mm"),
        ("Power:", "120V/240V AC, 50-60Hz"),
        ("Connectivity:", "USB 3.0, Ethernet, Wi-Fi 6"),
        ("Resolution:", "600 dpi optical, 1200 dpi interpolated"),
        ("Speed:", "40 pages per minute (simplex), 80 ipm (duplex)"),
    ]
    gt_parts = [title]
    for label, val in sections:
        gt_parts.append(f"{label} {val}")
    gt = "\n".join(gt_parts)

    img = Image.new("RGB", (650, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), title, fill="black", font=get_bold_font(20))
    draw.line([(40, 58), (600, 58)], fill="black", width=2)
    y = 75
    for label, val in sections:
        draw.text((40, y), label, fill="#333333", font=get_bold_font(13))
        draw.text((200, y), val, fill="black", font=get_font(13))
        y += 32
    save(img, "spec_sheet", gt)

# === 6. Small text / fine print ===
def gen_fine_print():
    text = (
        "TERMS AND CONDITIONS: By using this service, you agree to the following terms. "
        "All data submitted for processing is handled in accordance with our privacy policy. "
        "We do not store uploaded documents beyond the processing window (maximum 60 seconds). "
        "Results are returned via encrypted HTTPS connection. Service availability is provided "
        "on a best-effort basis with no guaranteed uptime SLA for free-tier accounts. "
        "Paid accounts receive 99.9% uptime guarantee. Refunds are available within 24 hours "
        "of purchase minus applicable platform fees. API rate limits apply per account tier."
    )
    img = Image.new("RGB", (700, 250), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(10)
    bold = get_bold_font(11)
    draw.text((30, 20), "TERMS AND CONDITIONS:", fill="black", font=bold)
    body = text.replace("TERMS AND CONDITIONS: ", "")
    wrapped = textwrap.fill(body, width=100)
    draw.text((30, 40), wrapped, fill="#333333", font=font)
    save(img, "fine_print", text)

# === 7. Code/technical text ===
def gen_code_text():
    code = '''import openocr

client = openocr.Client(api_key="sk-abc123")

result = client.ocr(
    engine="openocr/tesseract",
    image="invoice.png",
    options={"language": "en", "dpi": 300}
)

print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Pages: {result.page_count}")'''

    img = Image.new("RGB", (600, 380), "#1e1e1e")
    draw = ImageDraw.Draw(img)
    mono = get_font(13)  # Will use DejaVuSans which is close enough
    y = 20
    for line in code.split("\n"):
        color = "#569cd6" if line.startswith("import") else "#ce9178" if '"' in line else "#d4d4d4"
        if line.strip().startswith("#") or line.strip().startswith("print"):
            color = "#dcdcaa" if "print" in line else "#6a9955"
        draw.text((20, y), line, fill=color, font=mono)
        y += 22
    save(img, "code_snippet", code)


print("Generating benchmark images...")
gen_dense_paragraph()
gen_multi_column()
gen_table_form()
gen_noisy_scan()
gen_mixed_fonts()
gen_fine_print()
gen_code_text()
print(f"\nDone! Generated 7 new test images + ground truths.")
