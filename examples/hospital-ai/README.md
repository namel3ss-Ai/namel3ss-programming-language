# Hospital Support AI - Multi-Agent System

A comprehensive, secure hospital support application built with **Namel3ss (N3)** that provides AI-powered assistance for medical staff and patients.

## 🏥 Overview

This application demonstrates a production-ready multi-agent AI system for healthcare with:

- **AI-Powered Clinical Tools**: Triage assistance, medication lookup, care workflow planning
- **Role-Based Access Control**: Separate interfaces for clinicians, patients, and administrators
- **RAG-Enhanced Knowledge**: Grounded in medical guidelines and drug databases
- **Local-First Architecture**: Runs on self-hosted models for HIPAA compliance
- **Patient-Friendly Communication**: AI-assisted message translation

## 🎯 Key Features

### For Clinicians

1. **Triage Assistant** 
   - AI-powered triage recommendations based on symptoms and vitals
   - Emergency red flag detection
   - ESI acuity scoring
   - Evidence-based guidelines citations

2. **Medication Advisor**
   - Drug information and safety checks
   - Drug-drug interaction analysis
   - Patient-specific contraindications
   - Dosing guidance with organ function adjustments

3. **Care Workflow Assistant**
   - Evidence-based clinical pathways
   - Step-by-step care planning
   - Milestone tracking
   - Complication prevention

4. **Appointment Management**
   - AI-assisted scheduling
   - Slot optimization
   - Calendar integration

5. **Patient Messaging**
   - Secure communication
   - AI-powered message translation to patient-friendly language
   - Template support

### For Patients

1. **Appointment Viewing**
   - Upcoming appointments
   - Confirmation details
   - Add to calendar

2. **Secure Messaging**
   - Communicate with care team
   - View responses
   - Restricted to appropriate topics

3. **Health Education**
   - Personalized health information
   - Condition-specific materials
   - Medication information
   - Approved, vetted content

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    Hospital Support AI                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Triage     │  │ Medication   │  │  Workflow    │  │
│  │   Agent      │  │   Agent      │  │   Agent      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│         └──────────┬───────┴──────────────────┘          │
│                    │                                     │
│         ┌──────────▼────────────┐                       │
│         │   RAG Knowledge Base  │                       │
│         │ ┌──────────────────┐  │                       │
│         │ │Medical Guidelines│  │                       │
│         │ │  Drug Database   │  │                       │
│         │ │Clinical Pathways │  │                       │
│         │ └──────────────────┘  │                       │
│         └───────────────────────┘                       │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │   Booking    │  │  Messaging   │                    │
│  │   Agent      │  │   Agent      │                    │
│  └──────────────┘  └──────────────┘                    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Language**: Namel3ss (N3) - declarative AI application language
- **Local Models**: Ollama (Llama 3.1 8B) for HIPAA compliance
- **Embeddings**: Nomic Embed Text (local)
- **Vector DB**: PostgreSQL with pgvector
- **RAG**: Hybrid search with reranking
- **Frontend**: Auto-generated from N3 UI declarations
- **Backend**: FastAPI (auto-generated from N3)

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **PostgreSQL 15+** with pgvector extension
- **Ollama** (for local models)
- **Namel3ss CLI** (`npm install -g namel3ss`)

### Installation

1. **Clone the repository**
   ```bash
   cd examples/hospital-ai
   ```

2. **Install Ollama models**
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

3. **Setup PostgreSQL with pgvector**
   ```bash
   # Install pgvector extension
   psql -U postgres
   CREATE DATABASE hospital_ai;
   CREATE DATABASE hospital_ai_vectors;
   \c hospital_ai_vectors
   CREATE EXTENSION vector;
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Load sample data**
   ```bash
   # Create data directories
   mkdir -p data/{medical_guidelines,drugs,pathways,patient_education}
   
   # Copy sample data
   cp sample_data/* data/
   ```

6. **Build and run**
   ```bash
   # Compile Namel3ss to runnable application
   namel3ss build hospital_support_ai.ai
   
   # Initialize databases and load datasets
   namel3ss migrate
   namel3ss dataset:load medical_guidelines
   namel3ss dataset:load drug_database
   namel3ss dataset:load clinical_pathways
   
   # Start the application
   namel3ss run --port 8000
   ```

7. **Access the application**
   - Open browser to `http://localhost:8000`
   - Login page will appear
   - Use demo credentials (see below)

### Demo Credentials

For testing purposes, you can use:

**Clinician**
- Username: `dr.smith`
- Password: `demo123`
- Role: `clinician`

**Patient**
- Username: `patient.doe`
- Password: `demo123`
- Role: `patient`

**Admin**
- Username: `admin`
- Password: `admin123`
- Role: `admin`

## 📁 Project Structure

```
hospital-ai/
├── hospital_support_ai.ai       # Main app config, security, models
├── agents.ai                    # Agent definitions
├── datasets_rag.ai              # Dataset and RAG configs
├── ui_clinician.ai              # Clinician dashboard pages
├── ui_clinician_extended.ai     # Additional clinician pages
├── ui_patient.ai                # Patient dashboard pages
├── data/
│   ├── medical_guidelines/      # Triage protocols, clinical guidelines
│   ├── drugs/                   # Drug monographs, interactions
│   ├── pathways/                # Clinical care pathways
│   └── patient_education/       # Patient-facing materials
├── sample_data/                 # Sample datasets for testing
└── README.md                    # This file
```

## ✨ Declarative UI Innovation

This project showcases **HTML-free UI syntax** - a revolutionary approach where you define interfaces using pure configuration, eliminating the need to write HTML, divs, or CSS classes.

### Traditional vs. Declarative Approach

**❌ Old Way (HTML-heavy):**
```yaml
item_render: |
  <div class="appointment-card {{ status }}">
    <div class="header">
      <div class="type-badge {{ type }}">{{ type | humanize }}</div>
      <div class="status-badge {{ status }}">{{ status | humanize }}</div>
    </div>
    <div class="info">
      <div class="date"><icon:calendar/> {{ date }}</div>
      <div class="provider"><icon:user-md/> {{ provider }}</div>
    </div>
    {% if instructions %}
    <div class="instructions">
      <strong>Instructions:</strong> {{ instructions }}
    </div>
    {% endif %}
  </div>
```

**✅ New Way (Pure declarative):**
```yaml
item:
  type: card
  style: appointment
  state_class: "{{ status }}"
  
  header:
    badges:
      - field: type
        style: type_badge
        transform: humanize
      - field: status
        style: status_badge
        transform: humanize
        
  sections:
    - type: info_grid
      columns: 2
      items:
        - icon: calendar
          field: date
        - icon: user-md
          field: provider
          
    - type: text_section
      condition: "instructions != null"
      content:
        label: "Instructions:"
        text: "{{ instructions }}"
```

### Key Declarative Patterns

1. **Semantic Components**: Use `type: card`, `type: info_grid`, `type: message_bubble` instead of divs
2. **Field-Based Rendering**: `field: name` instead of `<div>{{ name }}</div>`
3. **Transform Chains**: `transform: {format: "MMMM DD", humanize: true}` instead of Jinja filters
4. **Conditional Sections**: `condition: "field != null"` instead of `{% if %}`
5. **State Classes**: `state_class: "{{ status }}"` for dynamic styling
6. **Icon Integration**: `icon: calendar` instead of `<icon:calendar/>`
7. **Automatic Layouts**: `type: info_grid, columns: 2` handles responsive layout

### Benefits

- **90% less code**: Example appointment cards went from 60 lines HTML → 25 lines config
- **No HTML knowledge required**: Pure data structure definitions
- **Type-safe**: Compiler validates field names and structure
- **Consistent styling**: Automatic application of theme and design system
- **Responsive by default**: Layouts adapt to screen sizes automatically
- **Accessible**: ARIA labels and semantic HTML generated automatically

## 🔒 Security & Compliance

### Role-Based Access Control

Three roles with strict permissions:

1. **Clinician**
   - Access all AI agents
   - View all patient data
   - Create appointments
   - Send messages
   - Access RAG knowledge bases

2. **Patient**
   - View own appointments only
   - View own messages only
   - Access approved health education
   - **Cannot** access AI agents
   - **Cannot** view other patients

3. **Admin**
   - All clinician permissions
   - Manage users
   - View system logs
   - Configure security settings

### HIPAA Considerations

- **Local models**: All AI inference runs on-premises
- **Data encryption**: HTTPS required in production
- **Audit logging**: All actions logged with user attribution
- **Access controls**: Role-based restrictions enforced at API and agent level
- **Session management**: Secure JWT tokens with timeout

### Security Features

- Role validation at every endpoint
- Agent-level permission checks
- Memory isolation per user/session
- Audit trails for all clinical actions
- Secure credential management

## 🔧 Configuration

### Environment Variables

Edit `.env` file:

```bash
# Model Provider
MODEL_PROVIDER=local              # or 'openai', 'anthropic'
LOCAL_MODEL_URL=http://localhost:11434
LOCAL_MODEL_NAME=llama3.1:8b

# Cloud Fallback (optional)
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Databases
DATABASE_URL=postgresql://localhost:5432/hospital_ai
VECTOR_DB_URL=postgresql://localhost:5432/hospital_ai_vectors

# RAG
EMBEDDING_MODEL=local
EMBEDDING_MODEL_NAME=nomic-embed-text
RERANKER_ENABLED=true

# Security
JWT_SECRET=change-this-in-production
REQUIRE_HTTPS=true
SESSION_TIMEOUT=3600

# Application
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development           # or 'staging', 'production'
```

### Switching to Cloud Models

To use OpenAI or Anthropic instead of local models:

```bash
# In .env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your-openai-key

# Or for Anthropic
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-key
```

The application will automatically use the cloud provider for reasoning while keeping embeddings local.

## 📊 Datasets

### Medical Guidelines

Place medical guideline documents in `data/medical_guidelines/`:

- Triage protocols (e.g., ESI, MTS)
- Emergency criteria
- Clinical assessment guidelines
- Evidence-based protocols

Supported formats: `.md`, `.pdf`, `.txt`

### Drug Database

Place drug information in `data/drugs/` as JSON files:

```json
{
  "drug_name": "Metformin",
  "generic_name": "Metformin HCl",
  "brand_names": ["Glucophage", "Fortamet"],
  "drug_class": "Biguanide",
  "indications": ["Type 2 Diabetes Mellitus"],
  "contraindications": ["Severe renal impairment", "Metabolic acidosis"],
  "dosing": {
    "adult_dose": "500-2550 mg daily in divided doses",
    "renal_adjustment": "Reduce dose if eGFR < 45"
  },
  "interactions": [
    {
      "drug": "Contrast dye",
      "severity": "major",
      "description": "Risk of lactic acidosis"
    }
  ]
}
```

### Clinical Pathways

Place pathway documents in `data/pathways/`:

- Condition-specific care pathways
- Treatment protocols
- Care transitions
- Discharge planning guides

### Patient Education

Place approved patient materials in `data/patient_education/`:

- Condition explanations
- Medication instructions
- Lifestyle guidance
- Preventive care information

## 🧪 Testing

### Test the Triage Agent

```bash
curl -X POST http://localhost:8000/api/agents/triage \
  -H "Authorization: Bearer <clinician-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P12345",
    "patient_age": 45,
    "patient_sex": "male",
    "symptoms": ["chest pain", "shortness of breath"],
    "symptom_duration": "1 hour",
    "symptom_severity": "severe",
    "temperature": 98.6,
    "heart_rate": 110,
    "blood_pressure": "150/95",
    "oxygen_saturation": 94
  }'
```

### Test the Medication Agent

```bash
curl -X POST http://localhost:8000/api/agents/medication \
  -H "Authorization: Bearer <clinician-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "interaction_check",
    "medication_name": "Warfarin",
    "patient_age": 65,
    "patient_sex": "female",
    "current_medications": ["Aspirin", "Metformin"],
    "allergies": ["Penicillin"],
    "chronic_conditions": ["Atrial fibrillation", "Diabetes"]
  }'
```

## 📈 Extending the System

### Adding New Agents

Create a new agent in `agents.ai`:

```n3
agent my_new_agent:
  name: "My Agent"
  description: "What it does"
  
  permissions:
    required_role: clinician
    
  model: local_reasoning
  
  inputs:
    field_name: type
    
  uses_rag: dataset_name
  
  structured_output:
    schema:
      result_field: type
      
  prompt: |
    Your agent prompt here
```

### Adding New UI Pages

Create pages in UI files:

```n3
page my_new_page:
  path: "/clinician/my-page"
  requires_role: clinician
  title: "My Page"
  
  ui:
    type: dashboard
    # ... page definition
```

### Adding New Datasets

1. Create dataset definition in `datasets_rag.ai`
2. Add RAG configuration
3. Place data files in appropriate directory
4. Load with `namel3ss dataset:load <dataset_name>`

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Additional clinical agents (billing, lab orders, etc.)
- [ ] More UI components and themes
- [ ] Enhanced mobile responsiveness
- [ ] Real EHR integrations
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Voice input for clinical documentation

## 📚 Learn More

- [Namel3ss Documentation](../../docs/INDEX.md)
- [Agent System Guide](../../docs/ai_features/AGENT_SYSTEM.md)
- [RAG Guide](../../docs/RAG_GUIDE.md)
- [Security Architecture](../../docs/SECURITY_ARCHITECTURE.md)
- [UI Components](../../docs/FRONTEND_INTEGRATION_MODES.md)

## ⚖️ License

This example is provided under the MIT License. See [LICENSE](../../LICENSE) for details.

## ⚠️ Disclaimer

**THIS IS A DEMONSTRATION APPLICATION**

This application is provided for **educational and demonstration purposes only**. It is **NOT intended for actual clinical use** without:

1. Proper medical validation and testing
2. HIPAA compliance audit
3. Clinical safety review
4. Regulatory approval where required
5. Integration with certified EHR systems
6. Professional liability coverage

**No Medical Advice**: This system does not provide medical advice, diagnosis, or treatment. All AI outputs are assistive only and require clinical judgment and verification.

**Not a Substitute for Professional Care**: Always consult qualified healthcare professionals for medical decisions.

## 📞 Support

For issues or questions about this example:

- Open an issue in the main repository
- Check the [Troubleshooting Guide](../../TROUBLESHOOTING.md)
- Join the Namel3ss community Discord

---

**Built with ❤️ using Namel3ss (N3) - The declarative AI application language**
