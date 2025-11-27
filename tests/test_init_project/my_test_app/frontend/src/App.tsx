import React from 'react'

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Namel3ss App
          </h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <div className="border-4 border-dashed border-gray-200 rounded-lg p-8">
              <h2 className="text-2xl font-semibold mb-4">
                Welcome to Your Namel3ss Application
              </h2>
              <p className="text-gray-600 mb-4">
                This is a starter frontend for your Namel3ss app.
              </p>
              <div className="space-y-2">
                <p className="text-sm text-gray-500">
                  📝 Edit <code className="bg-gray-200 px-2 py-1 rounded">app.ai</code> to modify your application
                </p>
                <p className="text-sm text-gray-500">
                  🔄 Run <code className="bg-gray-200 px-2 py-1 rounded">namel3ss build</code> to generate backend/frontend
                </p>
                <p className="text-sm text-gray-500">
                  🚀 Run <code className="bg-gray-200 px-2 py-1 rounded">namel3ss run</code> to start development server
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
