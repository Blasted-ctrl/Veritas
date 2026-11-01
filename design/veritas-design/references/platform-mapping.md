# Veritas platform mapping

## 1. HTML / CSS / WEB

Load the two self-hosted Hanken font files from assets/. The light mode is primary; honor the system preference and let the user override it.

```css
:root{--background:#eef1ec;--surface1:#fafbf8;--surface2:#e2e8df;--surface3:#c9d2c8;--border:#c9d2c8;--border-visible:#5c6c61;--text1:#203b34;--text2:#52635b;--text3:#5c6c61;--text4:#5c6c61;--accent:#416b4b;--accent-subtle:#e5eddc;--accent-ink:#fafbf8;}
[data-theme="dark"]{--background:#15231d;--surface1:#1d3025;--surface2:#203b34;--surface3:#2c4839;--border:#385344;--border-visible:#9cafa0;--text1:#fafbf8;--text2:#c9d2c8;--text3:#a6b8a9;--text4:#a6b8a9;--accent:#b9d0a5;--accent-subtle:#182c1c;--accent-ink:#182c1c;}
```

## 2. SWIFTUI / iOS

The shipping app is web-only. For a native adaptation, register the same OFL font files under UIAppFonts, map the semantic colors to named asset-catalog colors with light/dark variants, and use the native file importer.

```swift
extension Font {
  static let veritasBody = Font.custom("HankenGrotesk-Regular", size: 16)
  static let veritasHeading = Font.custom("HankenGrotesk-SemiBold", size: 32)
}
enum VeritasSpacing { static let small: CGFloat = 8; static let group: CGFloat = 16; static let section: CGFloat = 48 }
enum VeritasRadius { static let control: CGFloat = 8; static let panel: CGFloat = 16 }
```

## 3. REACT / TAILWIND

Use the existing native CSS custom properties. Components inherit Hanken from the root. Map utility names to semantic variables, never duplicate independent color literals. The production globals.css shortens text1 to --ink, text2 to --muted, text3 to --quiet, surface1 to --surface, and border to --line.

```js
export default { theme: { extend: { colors: { background: "var(--bg)", surface: "var(--surface)", ink: "var(--ink)", accent: "var(--accent)" }, borderRadius: { control: "8px", panel: "16px" } } } };
```
