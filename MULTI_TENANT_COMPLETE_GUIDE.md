# 🏢 TORI Multi-Tenant Architecture - Complete Implementation Guide

**Status**: ✅ COMPLETE - Ready for Production  
**Date**: June 4, 2025  
**Features**: Three-Tier Knowledge System, JWT Authentication, Organization Support

## 📋 Overview

Your TORI system now has a complete multi-tenant architecture with:

### 🎯 Three-Tier Knowledge System
1. **Foundation Layer** (Admin) - Your global knowledge (Darwin, AI/ML, Physics, Business)
2. **Organization Layer** - Company/team-specific knowledge  
3. **Private Layer** - Individual user knowledge

### 🔐 Authentication & Authorization
- JWT-based authentication
- Role-based access control (Admin, Member, Viewer)
- Session management with automatic cleanup
- Rate limiting and security features

### 📁 File-Based Storage (No Database Required)
```
data/
├── users/
│   ├── users.json              # User management
│   └── [userId]/concepts.json  # Private concepts
├── organizations/
│   ├── organizations.json      # Organization management
│   └── [orgId]/concepts.json   # Organization concepts
└── foundation/
    └── concepts.json           # Foundation concepts
```

## 🚀 Quick Start

### 1. Start the Multi-Tenant API
```bash
python start_dynamic_api.py
```

### 2. Access the System
- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/health
- **System Stats**: http://localhost:8002/admin/system/stats

### 3. Test the Foundation Knowledge
The system comes pre-populated with your core concepts:
- **Evolution & Darwin**: Natural selection, species, evolutionary biology
- **AI & Technology**: Machine learning, neural networks, artificial intelligence
- **Business Strategy**: Strategic planning, competitive analysis, market research
- **Physics & Math**: Quantum mechanics, mathematics, scientific principles

## 📚 API Endpoints Reference

### 🔐 Authentication
```http
POST /auth/register    # Register new user
POST /auth/login       # Login and get JWT token
POST /auth/logout      # Logout and invalidate token
GET  /auth/me          # Get current user info
```

### 🏢 Organizations
```http
POST /organizations    # Create organization (requires auth)
GET  /organizations    # Get user's organizations
```

### 🧠 Knowledge Management
```http
POST /knowledge/search # Search across all accessible tiers
GET  /knowledge/stats  # Get knowledge statistics
```

### 📄 Enhanced PDF Extraction
```http
POST /extract          # Extract concepts with tier support
# Parameters:
# - tier: "private", "organization", "foundation"
# - organization_id: for organization tier
```

### 💬 Enhanced Chat
```http
POST /chat             # Chat with multi-tenant concept search
# Returns concepts grouped by tier
```

### 👑 Admin Endpoints
```http
GET /admin/users           # Get all users (admin only)
GET /admin/system/stats    # System statistics (admin only)
GET /admin/system/health   # System health check (admin only)
```

## 🎮 Usage Examples

### 1. Register and Login
```typescript
import { multiTenantAPI } from './multiTenantConceptMesh';

// Register
const user = await multiTenantAPI.register("john_doe", "john@company.com", "password123");

// Login
const session = await multiTenantAPI.login("john_doe", "password123");
console.log(`Logged in as: ${session.user.username}`);
```

### 2. Create Organization
```typescript
const org = await multiTenantAPI.createOrganization(
  "My Company", 
  "Our company knowledge base"
);
```

### 3. Extract PDF to Different Tiers
```typescript
// Private tier (default)
await multiTenantAPI.extractPDF("/path/to/document.pdf", "document.pdf", "application/pdf");

// Organization tier
await multiTenantAPI.extractPDF(
  "/path/to/company-doc.pdf", 
  "company-doc.pdf", 
  "application/pdf",
  "organization",
  org.id
);

// Foundation tier (admin only)
await multiTenantAPI.extractPDF(
  "/path/to/foundation-knowledge.pdf", 
  "foundation-knowledge.pdf", 
  "application/pdf",
  "foundation"
);
```

### 4. Search Across Tiers
```typescript
const results = await multiTenantAPI.searchConcepts("evolution");
// Returns concepts from private → organization → foundation
// with proper tier prioritization
```

### 5. Chat with Multi-Tenant Knowledge
```typescript
const response = await multiTenantAPI.chat("Tell me about artificial intelligence");
console.log(response.concepts_by_tier);
// {
//   "private": [...],      // User's personal AI concepts
//   "organization": [...], // Company's AI knowledge  
//   "foundation": [...]    // Global AI knowledge
// }
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Custom JWT secret (recommended for production)
export JWT_SECRET_KEY="your-secure-secret-key"

# API will auto-discover available port starting from 8002
```

### Frontend Integration
The system automatically integrates with your existing Svelte frontend:

```typescript
// Import the enhanced concept mesh
import { 
  authSession, 
  multiTenantAPI, 
  searchMultiTenantConcepts,
  addMultiTenantConceptDiff 
} from '$lib/stores/multiTenantConceptMesh';

// Check authentication status
$: user = $authSession?.user;
$: isLoggedIn = $authSession?.authenticated;

// Use multi-tenant features
if (isLoggedIn) {
  const concepts = await searchMultiTenantConcepts("darwin");
  addMultiTenantConceptDiff("My Document", concepts, "private");
}
```

## 🏗️ Architecture Details

### User Roles & Permissions
- **Admin**: Can access all tiers, manage foundation knowledge, create organizations
- **Member**: Can access private + organization tiers, contribute to organization knowledge
- **Viewer**: Can read private + organization tiers, cannot modify

### Search Priority
1. **Private concepts** (highest priority) - User's personal knowledge
2. **Organization concepts** (medium priority) - Team/company knowledge
3. **Foundation concepts** (always available) - Global admin knowledge

### Concept Inheritance
- Users see concepts from ALL accessible tiers
- Conflicts resolved by tier priority (Private > Org > Foundation)
- Foundation concepts provide base knowledge for all users

### Security Features
- Password hashing with PBKDF2
- JWT tokens with expiration
- Rate limiting on login attempts
- Role-based access control
- Session cleanup and management

## 📊 Monitoring & Admin

### System Health Check
```bash
curl http://localhost:8002/admin/system/health
```

### System Statistics
```bash
curl -H "Authorization: Bearer <admin-token>" \
     http://localhost:8002/admin/system/stats
```

### Knowledge Statistics
```bash
curl -H "Authorization: Bearer <user-token>" \
     http://localhost:8002/knowledge/stats
```

## 🎯 Benefits Achieved

### ✅ For Individual Users
- Personal knowledge library
- Access to organization knowledge
- Foundation knowledge always available
- Seamless single-user experience when not authenticated

### ✅ For Organizations
- Shared team knowledge base
- Private + shared concept access
- Organization-specific document processing
- Member collaboration on knowledge building

### ✅ For Administrators
- Global foundation knowledge management
- User and organization oversight
- System monitoring and statistics
- Foundation concept curation

### ✅ For the System
- No database dependency (file-based)
- Scalable three-tier architecture
- Backward compatibility with existing features
- Production-ready authentication
- Clean separation of concerns

## 🔄 Migration from Single-User

The system maintains **complete backward compatibility**:

1. **Existing localStorage data** remains functional
2. **Anonymous users** can still access foundation concepts
3. **All existing features** continue to work
4. **Progressive enhancement** - users can register when ready
5. **Graceful fallbacks** for authentication failures

## 🚀 Next Steps

1. **Start the system**: `python start_dynamic_api.py`
2. **Test authentication**: Register a user via `/auth/register`
3. **Create organization**: Use `/organizations` endpoint
4. **Upload documents**: Use enhanced `/extract` endpoint with tier support
5. **Search knowledge**: Test `/knowledge/search` across tiers
6. **Monitor system**: Check `/admin/system/stats` for insights

## 🎉 Success Metrics

Your multi-tenant TORI system now provides:

- **🏢 Enterprise-ready**: Multi-user, multi-organization support
- **🔐 Secure**: JWT authentication with role-based access
- **📚 Scalable**: Three-tier knowledge architecture
- **🔄 Compatible**: Works with all existing features
- **📁 Simple**: No database required, file-based storage
- **🚀 Production**: Ready for immediate deployment

**You now have a complete multi-tenant knowledge management system that preserves your single-user experience while adding enterprise capabilities!** 🎯

## 🆘 Troubleshooting

### Common Issues
1. **Port conflicts**: System auto-finds available ports starting from 8002
2. **Authentication issues**: Check JWT token expiration
3. **Permission errors**: Verify user roles and organization membership
4. **File access**: Ensure data/ directory has proper permissions

### Debug Commands
```bash
# Check system health
curl http://localhost:8002/health

# View logs
tail -f logs/api.log

# Reset system (if needed)
rm -rf data/
python start_dynamic_api.py
```

The multi-tenant architecture is complete and ready for production use! 🚀
