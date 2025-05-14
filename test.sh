# Step 6: Set environment variables
echo "🔧 Setting Railway environment variables..."
railway variables --set "FLASK_SECRET_KEY=echelon_secret_$(openssl rand -hex 8)" 
railway variables --set "ALLOWED_ORIGINS=https://final-omega-drab.vercel.app,http://localhost:3000" 
railway variables --set "DEBUG=false"

# Step 7: Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Step 8: Display the deployment URL
echo "✅ Railway deployment complete!"
echo "📋 Getting deployment information..."
railway status

echo ""
echo "🔗 To integrate with your Vercel frontend:"
echo "1. Go to your Vercel dashboard: https://vercel.com/dashboard"
echo "2. Select your project 'final-omega-drab'"
echo "3. Go to 'Settings' > 'Environment Variables'"
echo "4. Add 'NEXT_PUBLIC_API_URL' with the Railway URL shown above"
echo "5. Redeploy your frontend"
echo ""
echo "Happy threat hunting with Echelon! 🕵️‍♀️"