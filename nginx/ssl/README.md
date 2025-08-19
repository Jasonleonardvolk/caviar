# SSL Certificates Directory

Place your SSL certificates here:

1. **cert.pem** - Your SSL certificate
2. **key.pem** - Your private key

## Generate Self-Signed Certificates (Development Only)

```bash
# Generate private key
openssl genrsa -out key.pem 2048

# Generate certificate signing request
openssl req -new -key key.pem -out csr.pem

# Generate self-signed certificate
openssl x509 -req -days 365 -in csr.pem -signkey key.pem -out cert.pem
```

## Production Certificates

For production, use certificates from:
- Let's Encrypt (free)
- Your domain registrar
- Commercial CA

## Security Notes

- Never commit certificates to version control
- Set proper file permissions (600)
- Rotate certificates regularly
- Use strong key sizes (2048-bit minimum)
