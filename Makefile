# TORI Holographic Display System - Build Pipeline
# Main Makefile for complete system build

FRONTEND := frontend
BACKEND := .
TOOLS := tools

# Default target
.PHONY: all
all: build

# Complete build pipeline
.PHONY: build
build: quilt-video

# Install all dependencies
.PHONY: install
install: install-frontend install-backend
	@echo "==> All dependencies installed"

.PHONY: install-frontend
install-frontend:
	@echo "==> Installing frontend dependencies"
	cd $(FRONTEND) && npm install

.PHONY: install-backend
install-backend:
	@echo "==> Installing backend dependencies"
	pip install -r requirements.txt

# Transcode PNG to KTX2
.PHONY: transcode-ktx2
transcode-ktx2:
	@echo "==> Transcoding PNG to KTX2"
	cd $(FRONTEND) && node scripts/transcode-ktx2.js || true

# Generate quilt manifests
.PHONY: quilt-manifests
quilt-manifests:
	@echo "==> Generating quilt manifests"
	cd $(FRONTEND) && node scripts/build-quilt-manifest.js --root public/assets/quilt

# Build quilt videos (AV1 IVF + MP4)
.PHONY: quilt-videos
quilt-videos:
	@echo "==> Building quilt videos (AV1 IVF + AVC MP4)"
	node $(TOOLS)/build_quilt_video.js --mp4 avc

# Complete quilt pipeline
.PHONY: quilt-video
quilt-video: transcode-ktx2 quilt-manifests quilt-videos web-build
	@echo "==> Quilt video pipeline complete"

# Frontend build
.PHONY: web-build
web-build:
	@echo "==> Building frontend with Vite"
	cd $(FRONTEND) && npm run build

# Backend build
.PHONY: backend-build
backend-build:
	@echo "==> Building Python wheel"
	python -m build

# Development servers
.PHONY: dev
dev:
	@echo "==> Starting development servers"
	cd $(FRONTEND) && npm run dev

.PHONY: dev-backend
dev-backend:
	@echo "==> Starting backend development server"
	python main.py

.PHONY: dev-all
dev-all:
	@echo "==> Starting all development servers"
	cd $(FRONTEND) && npm run start:tori

# Testing
.PHONY: test
test: test-frontend test-backend
	@echo "==> All tests passed"

.PHONY: test-frontend
test-frontend:
	@echo "==> Running frontend tests"
	cd $(FRONTEND) && npm test

.PHONY: test-backend
test-backend:
	@echo "==> Running backend tests"
	pytest tests/ -v

.PHONY: test-e2e
test-e2e:
	@echo "==> Running E2E tests"
	cd $(FRONTEND) && npm run test:e2e

# Validation
.PHONY: validate
validate: validate-wgsl lint
	@echo "==> Validation complete"

.PHONY: validate-wgsl
validate-wgsl:
	@echo "==> Validating WGSL shaders"
	cd $(FRONTEND) && npm run validate:wgsl

.PHONY: lint
lint:
	@echo "==> Running linters"
	cd $(FRONTEND) && npm run lint

# Cleaning
.PHONY: clean
clean:
	@echo "==> Cleaning build artifacts"
	rm -rf $(FRONTEND)/dist
	rm -rf $(FRONTEND)/node_modules
	rm -rf $(FRONTEND)/.svelte-kit
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: clean-cache
clean-cache:
	@echo "==> Clearing caches"
	rm -rf $(FRONTEND)/node_modules/.vite
	rm -rf $(FRONTEND)/node_modules/.cache
	rm -rf .pytest_cache

# Docker
.PHONY: docker-build
docker-build:
	@echo "==> Building Docker images"
	docker-compose build

.PHONY: docker-up
docker-up:
	@echo "==> Starting Docker containers"
	docker-compose up -d

.PHONY: docker-down
docker-down:
	@echo "==> Stopping Docker containers"
	docker-compose down

# Production
.PHONY: production
production: clean install build test
	@echo "==> Production build complete"

.PHONY: deploy
deploy: production
	@echo "==> Deploying to production"
	# Add deployment commands here

# Help
.PHONY: help
help:
	@echo "TORI Holographic Display System - Build Commands"
	@echo ""
	@echo "Main targets:"
	@echo "  make build          - Complete build pipeline"
	@echo "  make install        - Install all dependencies"
	@echo "  make dev            - Start development servers"
	@echo "  make test           - Run all tests"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Quilt pipeline:"
	@echo "  make transcode-ktx2 - Convert PNG to KTX2"
	@echo "  make quilt-manifests- Generate quilt manifests"
	@echo "  make quilt-videos   - Build AV1/MP4 videos"
	@echo "  make quilt-video    - Complete quilt pipeline"
	@echo ""
	@echo "Other targets:"
	@echo "  make validate       - Validate shaders and lint"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make production     - Production build with tests"
	@echo "  make help           - Show this help"
