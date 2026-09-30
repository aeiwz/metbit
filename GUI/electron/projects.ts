import { constants } from "node:fs";
import { copyFile, readFile, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join } from "node:path";

const EXT = ".metbit";
const MAX = 50; // ponytail: a flat recent list; add folders or tags if people keep hundreds of projects

export interface ProjectEntry {
  path: string;
  name: string;
  folder: string;
  openedAt: string;
  exists: boolean;
  size?: number;
  modifiedAt?: string;
}

interface Stored {
  path: string;
  openedAt: string;
}

/** A project file name without folder or extension: printable, no path characters, at most 120 characters. */
export function cleanName(raw: string): string {
  const name = raw.trim();
  if (!name) throw new Error("Enter a name");
  if (name.length > 120) throw new Error("Use at most 120 characters");
  if (/[<>:"/\\|?*\u0000-\u001f]/.test(name) || name.startsWith(".")) throw new Error('Names cannot contain < > : " / \\ | ? * or start with a dot');
  return name;
}

const exists = (path: string) =>
  stat(path).then(
    () => true,
    () => false,
  );

/** The saved and opened .metbit files, newest first, kept in a JSON file in the app's user data. */
export class ProjectRegistry {
  constructor(private file: string) {}

  private async read(): Promise<Stored[]> {
    try {
      const data = JSON.parse(await readFile(this.file, "utf8"));
      return Array.isArray(data) ? data.filter((d) => typeof d?.path === "string") : [];
    } catch {
      return []; // first run, or an unreadable list: start fresh rather than fail
    }
  }

  private write(items: Stored[]) {
    return writeFile(this.file, JSON.stringify(items.slice(0, MAX), null, 2));
  }

  private async known(path: string): Promise<Stored[]> {
    const items = await this.read();
    if (!items.some((i) => i.path === path)) throw new Error("That file is not in the project list");
    return items;
  }

  async list(): Promise<ProjectEntry[]> {
    return Promise.all(
      (await this.read()).map(async ({ path, openedAt }) => {
        const base = { path, openedAt, name: basename(path, extname(path)), folder: dirname(path) };
        try {
          const s = await stat(path);
          return { ...base, exists: true, size: s.size, modifiedAt: s.mtime.toISOString() };
        } catch {
          return { ...base, exists: false };
        }
      }),
    );
  }

  async add(path: string) {
    if (extname(path).toLowerCase() !== EXT) throw new Error(`Projects are ${EXT} files`);
    const rest = (await this.read()).filter((i) => i.path !== path);
    await this.write([{ path, openedAt: new Date().toISOString() }, ...rest]);
  }

  async remove(path: string) {
    await this.write((await this.read()).filter((i) => i.path !== path));
  }

  async rename(path: string, name: string): Promise<string> {
    const items = await this.known(path);
    const target = join(dirname(path), cleanName(name) + EXT);
    if (target === path) return path;
    if (await exists(target)) throw new Error(`${basename(target)} already exists in that folder`);
    await rename(path, target);
    await this.write(items.map((i) => (i.path === path ? { ...i, path: target } : i)));
    return target;
  }

  async duplicate(path: string): Promise<string> {
    await this.known(path);
    const stem = join(dirname(path), basename(path, extname(path)) + " copy");
    for (let n = 1; ; n++) {
      const target = `${stem}${n === 1 ? "" : ` ${n}`}${EXT}`;
      try {
        await copyFile(path, target, constants.COPYFILE_EXCL); // never overwrites
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code === "EEXIST") continue;
        throw e;
      }
      await this.add(target);
      return target;
    }
  }
}
