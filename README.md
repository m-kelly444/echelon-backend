# Echelon Dashboard Utilities

This project demonstrates how to use environment variables in the Echelon Dashboard application.

## Environment Variables

The application uses the following environment variables:

### `NEXT_PUBLIC_API_URL`

This is the base URL for the Echelon API. It's used in the frontend to make API requests.

Example: `https://echelon-api.example.com`

### `JWT_SECRET`

This is a secret key used to sign and verify JSON Web Tokens (JWTs) for authentication. It should be a strong, random string that is kept private.

You can generate a secure JWT secret using:

\`\`\`bash
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
\`\`\`

### `DATABASE_URL`

This is the connection string for your Neon PostgreSQL database. It's automatically added when you set up the Neon integration with Vercel.

## How Environment Variables Are Used

1. **API Client (`lib/api-client.ts`)**
   - Uses `NEXT_PUBLIC_API_URL` to make requests to the Echelon API

2. **Authentication (`lib/auth.ts`)**
   - Uses `JWT_SECRET` to sign and verify JWT tokens

3. **Database Connection (`lib/db.ts`)**
   - Uses `DATABASE_URL` to connect to the PostgreSQL database

## Security Considerations

- `NEXT_PUBLIC_API_URL` is prefixed with `NEXT_PUBLIC_` because it needs to be accessible in the browser
- `JWT_SECRET` and `DATABASE_URL` are server-side only and should never be exposed to the client

## Example Usage

The `components/dashboard/example-usage.tsx` file demonstrates how to use the API client to make authenticated requests to the Echelon API.
