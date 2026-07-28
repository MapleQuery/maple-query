/**
 * `@types/pdfmake` covers the main entry but not the standard-font
 * bundles under `build/standard-fonts/`, which the PDF export loads to
 * get a monospaced face for SQL without embedding a second TTF family.
 *
 * Each of those modules exports the same pair: an `.afm` metrics map to
 * merge into pdfmake's virtual file system, and the font descriptor to
 * merge into the document definition's `fonts` section.
 */
declare module "pdfmake/build/fonts/*.js" {
  interface RobotoBundle {
    vfs: Record<string, string>;
    fonts: Record<string, Record<string, string>>;
  }
  const bundle: RobotoBundle;
  export default bundle;
  export const vfs: RobotoBundle["vfs"];
  export const fonts: RobotoBundle["fonts"];
}

declare module "pdfmake/build/standard-fonts/*.js" {
  interface StandardFontBundle {
    /** `data/<Face>.afm` → base64 metrics. */
    vfs: Record<string, string>;
    /** Family name → { normal, bold, italics, bolditalics }. */
    fonts: Record<string, Record<string, string>>;
  }
  const bundle: StandardFontBundle;
  export default bundle;
  export const vfs: StandardFontBundle["vfs"];
  export const fonts: StandardFontBundle["fonts"];
}
