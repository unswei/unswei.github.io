# EICRL website

This site uses Hugo Extended and the FixIt theme. GitHub Pages deploys changes
automatically when they are pushed or merged into `main`.

## Editing a member profile

Carl’s profile is in `content/people/carl-cai.md` and appears at
`/people/carl-cai/`. Edit that Markdown file on GitHub or in a local checkout.
The fields between the `---` lines set the name, description, role, affiliation,
optional portrait and optional links. Write the biography and research details
below those fields using Markdown. Keep `type: "profile"` so the shared layout
continues to apply.

To add a portrait, upload it to `static/images/people/`, then set, for example:

```yaml
image: "/images/people/carl-cai.jpg"
```

To add contact or external profile links, replace `links: []` with a list:

```yaml
links:
  - label: "GitHub"
    url: "https://github.com/your-username"
  - label: "Email"
    url: "mailto:your.name@unsw.edu.au"
```

Leave `image` empty and `links: []` until those details are ready; the layout
omits them. Suggested biography sections are Research interests, Current
projects, Selected publications and Background. Include only relevant sections
with real content.

## Adding another member

Use the reusable profile template:

```sh
hugo new content --kind profile people/given-family.md
```

Alternatively, copy `content/people/carl-cai.md` into a new file in the same
directory and replace Carl’s details. Give each member a distinct filename.
Profiles use `layouts/profile/single.html` and share their styling in
`assets/css/_custom.scss`.

Fill in the profile, set `draft: false` when ready, and add a card under the
appropriate heading’s `people-grid` in `content/people/_index.md`. Follow Carl’s
card as an example: keep the name as plain text and add a separate Profile link
to `/people/given-family/`. The listing is edited separately;
creating a profile does not automatically add a card.

Use the shared "Student researchers" group for HDR, undergraduate, coursework
and other student researchers, including graduates. Label HDR students as
"HDR student" on the people overview. Use the specific degree (such as
"PhD student" or "MPhil student") on each individual profile.
The role label is sufficient for student cards; omit a repeated generic summary.

All overview portraits use the same 112 × 112 pixel display size. For a student
card with a photo, follow Olivia’s or Tongzhong’s `person-card-with-portrait`
example. A square JPEG around 400 × 400 pixels is sufficient for both the
overview and individual profile.

## Local preview and checks

Install Hugo Extended (the deployment currently uses 0.161.1), then initialise
the theme submodule and start a preview:

```sh
git submodule update --init --recursive
hugo server --buildDrafts
```

Before submitting changes, build the site and check its internal links:

```sh
hugo --gc --minify
python3 scripts/check-local-links.py public
```
