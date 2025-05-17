import { PrismaClient } from "@prisma/client"
import { hash } from "bcryptjs"

const prisma = new PrismaClient()

async function main() {
  // Create a default admin user
  const adminPassword = await hash("admin123", 10)

  await prisma.user.upsert({
    where: { email: "admin@echelon.com" },
    update: {},
    create: {
      email: "admin@echelon.com",
      name: "Admin User",
      password: adminPassword,
      role: "admin",
    },
  })

  console.log("Database has been seeded.")
}

main()
  .catch((e) => {
    console.error(e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
  })
