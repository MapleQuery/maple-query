// Any NEXT_PUBLIC_ variable is inlined into the browser bundle. The API
// token once shipped that way and had to be rotated; a public variable
// whose name reads like a secret now fails the build instead.
const SECRET_LIKE = /(TOKEN|SECRET|PASSWORD|PRIVATE|API_KEY)/;
const exposed = Object.keys(process.env).filter(
  (name) =>
    name.startsWith("NEXT_PUBLIC_") &&
    SECRET_LIKE.test(name) &&
    // PostHog's project key is public by design (it only ingests events).
    name !== "NEXT_PUBLIC_POSTHOG_KEY",
);
if (exposed.length > 0) {
  throw new Error(
    `Refusing to build: ${exposed.join(", ")} would be inlined into the client ` +
      "bundle. Use a server-only variable (no NEXT_PUBLIC_ prefix) read from a route handler.",
  );
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
};

module.exports = nextConfig;
