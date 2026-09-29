import type { Metadata } from 'next'
import { Geist } from 'next/font/google'
import './globals.css'

const geist = Geist({ subsets: ['latin'], variable: '--font-geist' })

export const metadata: Metadata = {
  title: "Farhan's WakaTime",
  description: 'Self-hosted coding time tracker',
  icons: { icon: '/favicon.svg' },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${geist.variable} font-sans bg-surface text-onsurface antialiased`}>
        {children}
      </body>
    </html>
  )
}
