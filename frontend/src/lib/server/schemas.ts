// Zod schemas for API input validation
import { z } from 'zod';

// Billing schemas
export const CheckoutSchema = z.object({
  planId: z.enum(['free', 'plus', 'pro'])
});

export const PortalSchema = z.object({
  customerId: z.string().min(1)
});

// Template schemas
export const ExportSchema = z.object({
  input: z.string().min(3),
  layout: z.enum(['grid', 'xyz']).default('grid'),
  scale: z.coerce.number().min(0.01).max(10).default(0.12),
  zip: z.boolean().optional()
});

export const UploadSchema = z.object({
  mode: z.enum(['auto', 'glb', 'concept']).optional(),
  name: z.string().max(255).optional(),
  description: z.string().max(1000).optional(),
  tags: z.string().optional(),
  layout: z.enum(['grid', 'xyz']).optional(),
  scale: z.string().optional()
});

// File upload constraints
export const FILE_UPLOAD_LIMITS = {
  maxSize: 25 * 1024 * 1024, // 25MB
  allowedExtensions: ['.glb', '.json'],
  allowedMimeTypes: ['model/gltf-binary', 'application/json', 'application/octet-stream']
};

// Validate file upload
export function validateFileUpload(file: File): { valid: boolean; error?: string } {
  if (!file) {
    return { valid: false, error: 'No file provided' };
  }

  if (file.size > FILE_UPLOAD_LIMITS.maxSize) {
    return { valid: false, error: `File too large. Maximum size is ${FILE_UPLOAD_LIMITS.maxSize / 1024 / 1024}MB` };
  }

  const fileName = file.name.toLowerCase();
  const hasValidExtension = FILE_UPLOAD_LIMITS.allowedExtensions.some(ext => fileName.endsWith(ext));
  
  if (!hasValidExtension) {
    return { valid: false, error: `Invalid file type. Allowed: ${FILE_UPLOAD_LIMITS.allowedExtensions.join(', ')}` };
  }

  return { valid: true };
}