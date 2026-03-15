"""Generate VLM training data: synthetic screenshots + Gemini labels.

For each example:
1. Generate a screenshot image via Gemini 3.1 Flash image preview (OpenRouter)
2. Label it with Gemini 3 Flash using the extraction prompt from context_extractor.py
3. Save image + labels to vlm_training/

Uses ThreadPoolExecutor for parallel API calls.
"""

import base64
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen")

OPENROUTER_KEY = os.environ["OPENROUTER_KEY"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "vlm_training"
IMAGES_DIR = OUTPUT_DIR / "images"
DATA_PATH = OUTPUT_DIR / "training_data.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency: how many image+label pipelines to run in parallel
MAX_WORKERS = 6

EXTRACT_PROMPT = (
    "Focus on the MAIN CONTENT area of the screen (ignore browser tabs, "
    "bookmarks bar, and navigation chrome). Read all text in the main "
    "content: headings, paragraphs, labels, form fields, cards, lists, "
    "sidebar content, and dialog text. "
    "Extract: email addresses, URLs, people names, company names, product "
    "names, medical terms, acronyms, abbreviations, technical jargon, "
    "project names, phone numbers, addresses, and proper nouns. "
    "Include full phrases when they contain specialized vocabulary. "
    "Output ONLY a comma-separated list. Be exhaustive."
)

# --- Prompt categories ---

CATEGORIES = {
    "corporate_email": [
        "un correo de Gmail corporativo sobre una guía médica de cardiología con términos como ECG, troponina, Dr. Martínez",
        "an Outlook email thread about a patent filing for NeuralSync AI, with legal terms and attorney names",
        "un correo de Gmail sobre un reporte financiero Q3 2024 de Banco Santander con métricas EBITDA y ROI",
        "an Outlook email about a clinical trial update for Pfizer's mRNA-4157 vaccine candidate",
        "un correo corporativo de Gmail discutiendo la migración a AWS us-east-1 con referencias a Terraform y CloudFormation",
        "an email thread in Outlook about HIPAA compliance review for MedTech Solutions patient portal",
        "un correo de Gmail sobre la reunión del comité de ética del Hospital Universitario La Paz",
        "an Outlook email about Deloitte's SOX audit findings for FiscalYear 2024",
        "a Gmail corporate email discussing the merger between CrowdStrike and Humio with SEC filing references",
        "un correo de Outlook sobre el lanzamiento de la plataforma LATAM de MercadoLibre con integración Stripe",
        "an email about scheduling a cardiothoracic surgery consultation with Dr. Patel at Mount Sinai",
        "un correo sobre el proyecto MINERVA de inteligencia artificial del CSIC con referencias a transformers y BERT",
        "an Outlook email about intellectual property licensing for Qualcomm's Snapdragon X Elite chipset",
        "a Gmail thread about the FDA 510(k) submission for BioGenesis orthopedic implant",
        "un correo corporativo sobre la implementación de SAP S/4HANA en Repsol con módulos FI-CO y MM",
        "an email discussing the Kubernetes v1.29 migration plan with ArgoCD and Istio service mesh",
        "un correo de Gmail sobre el caso judicial García vs. Ministerio de Trabajo ref. 2024/TC/1847",
        "an Outlook email about the Phase III results of Novartis's Kisqali for HR+ breast cancer",
        "a corporate email about the JPMorgan Chase Q4 earnings call with EPS and P/E ratio discussion",
        "un correo sobre la renovación del contrato con Telefónica para servicios 5G NSA en la red MásOrange",
        "an email thread about the cybersecurity incident response for SolarWinds Orion vulnerability CVE-2024-38094",
        "a Gmail email about onboarding at McKinsey & Company with references to MECE framework and OKRs",
        "un correo sobre el protocolo de ensayo clínico del Hospital Clínic de Barcelona para CAR-T therapy",
        "an Outlook email about the Boeing 737 MAX MCAS software recertification with FAA references",
        "a corporate Gmail about the Series B funding round for Anthropic with term sheet from Spark Capital",
        "un correo de Gmail corporativo sobre prescripción de Metformina 850mg y revisión de HbA1c del paciente",
        "an Outlook email about GDPR data processing agreement with Snowflake for EU customer data",
        "a Gmail email from HR about the 401(k) Roth conversion deadline with Fidelity BrokerageLink details",
        "un correo sobre la licitación pública del Ayuntamiento de Madrid para infraestructura IoT smart city",
        "an email about the Gartner Magic Quadrant positioning for Datadog's observability platform",
    ],
    "chat_apps": [
        "una conversación de Slack sobre el lanzamiento del producto Kubernetes v1.28 con @carlos y @priya",
        "a WhatsApp group chat about planning a trip to Machu Picchu with flight details on LATAM Airlines",
        "a Microsoft Teams thread about the sprint retrospective for Project Phoenix with Jira tickets PHOE-234",
        "una conversación de Slack en #backend-team sobre un bug en el endpoint /api/v2/payments con stack trace",
        "a WhatsApp chat discussing the Real Madrid vs Barcelona match with score predictions",
        "a Slack conversation in #devops about Datadog alerts for prod-us-east-1 memory spikes",
        "una conversación de Teams sobre la presentación para el cliente Accenture con deadlines Q1 2025",
        "a Slack thread about the PostgreSQL 16 migration with pg_dump and pg_restore commands",
        "a WhatsApp group chat planning a birthday party at Nobu restaurant for @Jessica",
        "una conversación de Slack sobre el deploy de la versión 3.4.1 en el cluster EKS con Helm charts",
        "a Teams meeting chat about the UX review of the Figma prototype for the checkout redesign",
        "a Slack conversation about CircleCI pipeline failures in the monorepo with npm audit findings",
        "una conversación de WhatsApp sobre citas médicas en el Centro de Salud San Isidro con la Dra. López",
        "a Slack thread in #ml-team about fine-tuning LLaMA 3.1 with LoRA on 8x A100 GPUs",
        "a Teams chat about the Salesforce CPQ configuration for Enterprise tier pricing",
        "una conversación de Slack sobre el incidente P1 en el servicio de autenticación OAuth2 con Okta",
        "a WhatsApp conversation about university enrollment at MIT with course codes 6.036 and 18.065",
        "a Slack DM about reviewing the PR #4521 for the React Native navigation refactor",
        "a Teams thread about the Confluence documentation for the AWS Lambda migration playbook",
        "una conversación de Slack sobre la configuración de Grafana dashboards para métricas de Prometheus",
        "a WhatsApp chat about ordering supplies from McMaster-Carr part number 91251A546",
        "a Slack conversation about the Terraform state lock issue in the staging environment s3://tf-state-staging",
        "una conversación de Teams sobre el workshop de Design Thinking con la consultora IDEO",
        "a Slack thread about the Redis Cluster failover and Sentinel configuration for cache-prod-01",
        "a WhatsApp group chat about a recipe for Pad Thai with ingredients from Whole Foods Market",
        "a Slack conversation about the Vercel deployment preview for the Next.js 14 app with ISR issues",
        "una conversación de Slack sobre la integración con la API de Mercado Pago para pagos en ARS y MXN",
        "a Teams thread about the SCRUM standup with velocity metrics and burndown chart for Sprint 47",
        "a Slack DM about debugging the WebSocket connection drop in the Socket.io v4 implementation",
        "una conversación de WhatsApp sobre el pedido de Amazon con tracking AMZN-2024-XK7834 y entrega Prime",
    ],
    "code_editors": [
        "VS Code with a Python FastAPI project open showing routes for /api/v1/users with Pydantic models",
        "Xcode with a SwiftUI project showing ContentView.swift with NavigationStack and @Observable macro",
        "VS Code with a TypeScript Next.js 14 app router showing page.tsx with server components",
        "un editor VS Code con un proyecto Django mostrando models.py con campos ForeignKey y ManyToManyField",
        "Xcode showing a Metal shader file with kernel functions for GPU-accelerated image processing",
        "VS Code with a Rust project showing Cargo.toml and main.rs with tokio async runtime",
        "an IntelliJ IDEA window with a Spring Boot Java project showing @RestController and @Autowired annotations",
        "VS Code with a Go project showing a gRPC service definition in .proto files and generated stubs",
        "un editor VS Code con un proyecto React mostrando App.tsx con hooks useState y useEffect",
        "VS Code with a Kubernetes YAML manifest showing Deployment, Service, and Ingress resources",
        "Xcode with a CoreML model integration showing Vision framework request handlers",
        "VS Code with a Python MLX project showing model loading with mx.load and nn.Module subclass",
        "an editor showing a Dockerfile with multi-stage build for a Node.js application with nginx",
        "VS Code with a Terraform project showing aws_instance and aws_security_group resources",
        "un editor con un archivo SQL mostrando queries con JOIN, GROUP BY y window functions ROW_NUMBER()",
        "VS Code with a Flutter/Dart project showing a StatefulWidget with BLoC pattern implementation",
        "Xcode showing an ARKit project with ARSCNView and plane detection configuration",
        "VS Code with a C++ project showing CMakeLists.txt and a class using std::shared_ptr and std::vector",
        "an editor showing a GraphQL schema with type definitions, mutations, and resolvers",
        "VS Code with a Python project showing pytest fixtures and parametrize decorators",
        "un editor VS Code con un proyecto Vue.js mostrando composables con ref() y computed()",
        "VS Code showing a .github/workflows/ci.yml GitHub Actions workflow with matrix strategy",
        "Xcode with a WidgetKit extension showing TimelineProvider and Widget configuration",
        "VS Code with an Elixir/Phoenix project showing LiveView module with handle_event callbacks",
        "an editor showing a Prisma schema with model definitions and relation fields",
        "VS Code with a Python pandas/numpy script showing DataFrame operations and matplotlib plots",
        "un editor mostrando un archivo de configuración nginx.conf con upstream y location blocks",
        "VS Code with a Svelte project showing reactive declarations and component props",
        "an editor showing a Redis Lua script for rate limiting with EVALSHA command",
        "VS Code with a Python project showing SQLAlchemy ORM models with relationship() and backref",
    ],
    "documents": [
        "a Google Docs document showing a quarterly business review for Stripe with GMV metrics",
        "un documento Word con un contrato de arrendamiento entre Inmobiliaria García S.L. y el inquilino",
        "a PDF viewer showing a research paper about transformer architectures with citations to Vaswani et al.",
        "a Google Docs showing meeting notes from the Board of Directors at Tesla with agenda items",
        "un documento Word con un informe médico del Hospital Ramón y Cajal con diagnóstico y tratamiento",
        "a Notion page showing a product roadmap for Q1 2025 with epics and milestones",
        "a Google Docs with an academic thesis proposal on quantum computing with references to Shor's algorithm",
        "un PDF con las condiciones generales de un seguro de Mapfre con cláusulas de cobertura",
        "a Word document showing a technical specification for REST API design with OpenAPI 3.0 schemas",
        "a Google Docs showing a marketing campaign brief for Nike's Air Max 2025 launch",
        "un documento de Google Docs con actas de la junta de propietarios de la Comunidad Residencial Los Olivos",
        "a PDF showing a patent application for a neural network architecture by Google DeepMind",
        "a Word document with a due diligence report for the acquisition of Figma by Adobe",
        "un informe en Google Docs sobre evaluación de riesgos laborales según normativa ISO 45001",
        "a Notion page showing sprint planning documentation with user stories and acceptance criteria",
        "a PDF viewer showing a pharmaceutical prescribing information leaflet for Ozempic (semaglutide)",
        "a Google Docs document with a press release about SpaceX Starship SN28 orbital test flight",
        "un documento Word con un dictamen pericial forense para el caso 2024/PC/8923 del Juzgado de Madrid",
        "a LaTeX document in Overleaf showing a mathematics paper with equations and theorem environments",
        "a Google Docs showing an employee handbook for Shopify with PTO and benefits policies",
        "un PDF con el boletín oficial del Estado (BOE) mostrando la Ley Orgánica 3/2024",
        "a Word document showing a venture capital term sheet from Sequoia for a $50M Series B",
        "a Google Docs with interview questions and rubric for a Senior SRE position at Netflix",
        "un documento mostrando el plan de estudios de Ingeniería Informática de la Universidad Politécnica",
        "a PDF showing the NIST Cybersecurity Framework 2.0 assessment report for a healthcare organization",
        "a Notion document showing API documentation with endpoint descriptions and example payloads",
        "un documento de Google Docs con la propuesta de proyecto para el programa Horizonte Europa",
        "a Word document with a clinical protocol for a Phase II oncology trial with RECIST criteria",
        "a Google Docs showing a content calendar for TechCrunch with article assignments and deadlines",
        "a PDF showing an architecture decision record (ADR) for microservices migration with C4 diagrams",
    ],
    "browsers": [
        "Chrome showing a Stack Overflow question about React useCallback vs useMemo with code examples",
        "Safari showing the Apple Developer documentation for SwiftUI's @Environment property wrapper",
        "un navegador Chrome mostrando la página de Wikipedia sobre la historia del Real Madrid CF",
        "Chrome showing the AWS Console EC2 dashboard with running instances in us-west-2 region",
        "Safari showing a Medium article about fine-tuning LLMs with QLoRA and bitsandbytes",
        "Chrome showing GitHub repository page for facebook/react with open issues and pull requests",
        "un navegador mostrando la web de El País con un artículo sobre la economía española y el IBEX 35",
        "Chrome showing the Vercel dashboard with deployment logs and environment variables",
        "Safari showing the MDN Web Docs page for the Intersection Observer API with examples",
        "Chrome showing a Jira board with Kanban columns and tickets for the Platform team sprint",
        "un navegador Chrome mostrando MercadoLibre con productos de electrónica y precios en pesos",
        "Chrome showing the Grafana dashboard with Prometheus metrics for API latency p99",
        "Safari showing the Hacker News front page with articles about Rust, WebAssembly, and AI",
        "Chrome showing Google Cloud Console with BigQuery datasets and scheduled queries",
        "un navegador mostrando la página de reservas de Booking.com para hoteles en Barcelona",
        "Chrome showing the Stripe dashboard with payment intents, webhooks, and API keys",
        "Safari showing the PyPI page for the transformers library version 4.45.0 with dependencies",
        "Chrome showing a Confluence wiki page about the incident postmortem for the 2024-03-15 outage",
        "un navegador mostrando la web del Ministerio de Hacienda con información sobre el IRPF",
        "Chrome showing the Figma design system with component library and design tokens",
        "Safari showing the OpenAI API documentation for the chat completions endpoint",
        "Chrome showing Amazon.com product page for MacBook Pro M4 with specs and reviews",
        "un navegador Chrome mostrando LinkedIn con el perfil de un ingeniero de software senior",
        "Chrome showing the Docker Hub page for the official postgres:16-alpine image with tags",
        "Safari showing the Tailwind CSS v4 documentation with utility class examples",
        "Chrome showing a Google Analytics 4 dashboard with conversion funnels and user segments",
        "un navegador mostrando la web de Renfe con horarios del AVE Madrid-Barcelona",
        "Chrome showing the Terraform Registry page for the AWS provider with module documentation",
        "Safari showing the Ray Wenderlich tutorial on CoreData with SwiftUI integration",
        "Chrome showing the New Relic APM dashboard with transaction traces and error analytics",
    ],
    "spreadsheets": [
        "Google Sheets showing a financial model with DCF analysis and WACC calculations for a startup",
        "una hoja de Excel con el presupuesto anual del departamento de marketing con partidas y KPIs",
        "Excel showing a sales pipeline tracker with company names, deal stages, and ARR values",
        "Google Sheets with an employee roster showing names, roles, departments, and salary bands",
        "una hoja de cálculo con métricas de rendimiento del equipo de ventas con comisiones y cuotas",
        "Excel showing a project timeline Gantt chart with milestones and resource allocation",
        "Google Sheets with A/B test results showing conversion rates, p-values, and confidence intervals",
        "una hoja Excel con inventario de productos mostrando SKUs, precios y niveles de stock",
        "Excel showing a cap table with investor names, share classes, and ownership percentages",
        "Google Sheets with OKR tracking showing objectives, key results, and progress percentages",
        "una hoja de cálculo con datos de pacientes del ensayo clínico con valores de hemoglobina y peso",
        "Excel showing a vendor comparison matrix with pricing tiers and feature scores",
        "Google Sheets with website analytics data showing bounce rate, session duration, and page views",
        "una hoja Excel con la contabilidad de una PYME mostrando facturas, IVA y retenciones IRPF",
        "Excel showing a risk assessment matrix with likelihood, impact, and mitigation strategies",
        "Google Sheets with a content performance tracker showing engagement metrics and CTR",
        "una hoja de cálculo con datos meteorológicos de estaciones AEMET con temperatura y precipitación",
        "Excel showing a supply chain tracking sheet with PO numbers, suppliers, and lead times",
        "Google Sheets with customer feedback survey results showing NPS scores and verbatim comments",
        "una hoja Excel con el cuadro de amortización de un préstamo hipotecario a 30 años",
        "Excel showing a sprint velocity chart with story points completed per iteration",
        "Google Sheets with a recruitment pipeline showing candidates, interview stages, and ratings",
        "una hoja de cálculo con las calificaciones de estudiantes de la asignatura Bases de Datos",
        "Excel showing a marketing budget allocation with CPM, CPC, and ROAS by channel",
        "Google Sheets with cryptocurrency portfolio tracking showing BTC, ETH prices and P&L",
        "una hoja Excel con el registro de horas trabajadas y cálculo de nómina con deducciones SS",
        "Excel showing an inventory management sheet with reorder points and safety stock levels",
        "Google Sheets with SaaS metrics dashboard showing MRR, churn rate, and LTV:CAC ratio",
        "una hoja de cálculo con resultados de laboratorio clínico mostrando valores de glucosa y colesterol",
        "Excel showing a real estate comparables analysis with price per sqft and cap rates",
    ],
    "mixed": [
        "a macOS Terminal showing htop with process list and a running Docker container build",
        "Finder window showing a project directory structure with Python files and config folders",
        "un calendario de Google Calendar mostrando reuniones de la semana con participantes y salas",
        "macOS System Settings showing the Network preferences with Wi-Fi and VPN configurations",
        "a Terminal window with git log showing recent commits and branch history",
        "Finder showing the Applications folder with developer tools like Xcode, Docker, and iTerm",
        "un escritorio macOS con múltiples ventanas: Terminal, Chrome y VS Code side by side",
        "macOS Activity Monitor showing memory usage with processes like kernel_task and WindowServer",
        "a Terminal running a Python script with tqdm progress bar and training loss output",
        "Apple Notes showing a meeting agenda with action items and @mentions",
        "un Finder mostrando la carpeta Descargas con archivos PDF, XLSX y documentos varios",
        "macOS Keychain Access showing saved passwords and certificates for development",
        "a Terminal with kubectl commands showing pod status in a Kubernetes cluster",
        "Preview.app showing a PDF invoice from AWS for cloud services with line items",
        "un calendario de Outlook mostrando la agenda del día con videoconferencias de Zoom y Teams",
        "macOS Disk Utility showing APFS volumes and storage allocation",
        "a Terminal running brew install commands with package dependency resolution",
        "Spotlight search results showing files, applications, and web suggestions",
        "un escritorio macOS con la app Notas mostrando una lista de compras del supermercado Mercadona",
        "macOS Mail app showing an inbox with emails from Apple Developer, GitHub, and AWS",
        "a Terminal window showing SSH connection to a remote server with system stats",
        "Finder showing a Downloads folder with DMG installers and ZIP archives",
        "un Time Machine backup en progreso mostrando el estado de la copia de seguridad",
        "macOS Shortcuts app showing an automation workflow with actions",
        "a Terminal running a Jest test suite with pass/fail results and coverage report",
        "Preview.app showing a scanned document with OCR text recognition results",
        "un Finder mostrando un disco externo con carpetas de proyecto y archivos de base de datos",
        "macOS Console.app showing system logs with process IDs and log levels",
        "a Terminal with docker-compose up output showing multiple service startup logs",
        "Xcode Organizer showing app archives and TestFlight build distributions",
    ],
}


def call_openrouter(model: str, messages: list, max_tokens: int = 1024) -> dict:
    """Call OpenRouter API with retry on transient errors."""
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("Rate limited (429), waiting %ds before retry %d/3", wait, attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** attempt * 3
                log.warning("Server error %d, waiting %ds before retry %d/3", resp.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.warning("Request timeout, retry %d/3", attempt + 1)
            continue
        except requests.exceptions.ConnectionError as e:
            log.warning("Connection error: %s, retry %d/3", e, attempt + 1)
            time.sleep(2)
            continue
    raise RuntimeError(f"Failed after 3 retries for model {model}")


def generate_image(prompt: str) -> bytes | None:
    """Generate a screenshot image via Gemini 3.1 Flash image preview."""
    full_prompt = f"{prompt}. Formato: Screenshot mac, only screen, realistic, high resolution"

    try:
        result = call_openrouter(
            model="google/gemini-3.1-flash-image-preview",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=4096,
        )
    except Exception as e:
        log.error("Image gen failed: %s", e)
        return None

    choices = result.get("choices", [])
    if not choices:
        log.warning("No choices in image gen response")
        return None

    message = choices[0].get("message", {})

    # OpenRouter returns images in msg["images"] field
    images = message.get("images", [])
    if images:
        img_data = images[0]
        if isinstance(img_data, dict) and "image_url" in img_data:
            url = img_data["image_url"]["url"]
            if url.startswith("data:image"):
                b64 = url.split(",", 1)[1]
                return base64.b64decode(b64)
        elif isinstance(img_data, str):
            return base64.b64decode(img_data)

    # Fallback: check content for inline images
    content = message.get("content", "")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    b64 = url.split(",", 1)[1]
                    return base64.b64decode(b64)

    log.warning("No image found in response (keys: %s)", list(message.keys()))
    return None


def label_image(image_path: str) -> str | None:
    """Label a screenshot image using Gemini 3 Flash."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    try:
        result = call_openrouter(
            model="google/gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": EXTRACT_PROMPT},
                    ],
                }
            ],
            max_tokens=1024,
        )
    except Exception as e:
        log.error("Labeling failed for %s: %s", image_path, e)
        return None

    choices = result.get("choices", [])
    if not choices:
        log.warning("No choices in labeling response for %s", image_path)
        return None

    text = choices[0].get("message", {}).get("content", "")
    if isinstance(text, list):
        text = " ".join(
            p.get("text", "") for p in text
            if isinstance(p, dict) and p.get("type") == "text"
        )

    if not text or not text.strip():
        log.warning("Empty labels for %s", image_path)
        return None

    return text.strip()


def process_one(idx: int, category: str, prompt: str) -> dict | None:
    """Generate image + label for a single prompt. Returns entry dict or None."""
    tag = f"[{idx:04d}|{category}]"

    # Step 1: Generate image
    img_bytes = generate_image(prompt)
    if img_bytes is None:
        log.warning("%s SKIP — image generation failed", tag)
        return None

    img_path = IMAGES_DIR / f"{idx:04d}.png"
    img_path.write_bytes(img_bytes)
    log.info("%s image saved (%d KB)", tag, len(img_bytes) // 1024)

    # Step 2: Label image
    labels = label_image(str(img_path))
    if labels is None:
        log.warning("%s SKIP — labeling failed, removing image", tag)
        img_path.unlink(missing_ok=True)
        return None

    log.info("%s labeled (%d chars): %.80s...", tag, len(labels), labels)

    return {
        "index": idx,
        "category": category,
        "prompt": prompt,
        "image": f"images/{idx:04d}.png",
        "labels": labels,
    }


# Thread-safe data saver
_data_lock = threading.Lock()


def main():
    # Load existing data
    existing = []
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            existing = json.load(f)

    start_idx = len(existing)
    log.info("Resuming from index %d (%d existing examples)", start_idx, start_idx)

    # Build prompt list
    all_prompts = []
    for category, prompts in CATEGORIES.items():
        for prompt in prompts:
            all_prompts.append((category, prompt))

    random.seed(42)
    random.shuffle(all_prompts)

    remaining = all_prompts[start_idx:]
    total = len(all_prompts)
    log.info("Total prompts: %d, remaining: %d, workers: %d", total, len(remaining), MAX_WORKERS)

    data = list(existing)
    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for i, (category, prompt) in enumerate(remaining):
            idx = start_idx + i
            fut = pool.submit(process_one, idx, category, prompt)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:
                log.error("[%04d] Unexpected error: %s", idx, e)
                failed += 1
                continue

            if entry is None:
                failed += 1
            else:
                with _data_lock:
                    data.append(entry)
                    success += 1
                    # Save every 5 successes
                    if success % 5 == 0:
                        with open(DATA_PATH, "w") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                done = success + failed
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                log.info(
                    "Progress: %d/%d done (ok=%d fail=%d) — %.1f/min — %.0fs elapsed",
                    done, len(remaining), success, failed, rate, elapsed,
                )

    # Final save
    # Sort by index for consistency
    data.sort(key=lambda x: x["index"])
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(
        "DONE. %d successful, %d failed out of %d. Total: %d examples. Time: %.0fs (%.1f/min)",
        success, failed, len(remaining), len(data), elapsed, (success + failed) / elapsed * 60,
    )


if __name__ == "__main__":
    main()
