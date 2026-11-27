from pathlib import Path

from document_lab.paper.output import build_index, write_paper_output, write_section_files

EN_DASH = chr(0x2013)  # the dash a standard's caption prints: "Figure 1 - Title" with an en dash
LONG_BODY = 'Long body text that passes all thresholds. ' * 20


class TestWriteSectionFiles:
    def test_creates_md_files(self, tmp_path: Path):
        sections = [
            ('## Introduction', 'Intro body text.'),
            ('## Methods', 'Methods body text.'),
            ('## Results', 'Results body text.'),
        ]
        md_dir = tmp_path / 'md'
        info = write_section_files(sections, md_dir)

        assert len(info) == 3
        assert (md_dir / '01-introduction.md').exists()
        assert (md_dir / '02-methods.md').exists()
        assert (md_dir / '03-results.md').exists()

    def test_clears_existing_dir(self, tmp_path: Path):
        sections = [('## Intro', 'Body.')]
        md_dir = tmp_path / 'md'
        md_dir.mkdir(parents=True)
        (md_dir / 'old-file.md').write_text('stale')

        write_section_files(sections, md_dir)

        assert not (md_dir / 'old-file.md').exists()
        assert (md_dir / '01-intro.md').exists()

    def test_empty_heading_name(self, tmp_path: Path):
        sections = [('', 'Preamble body text.')]
        md_dir = tmp_path / 'md'
        info = write_section_files(sections, md_dir)
        assert info[0]['slug'] == 'body'
        assert (md_dir / '01-body.md').exists()

    def test_info_structure(self, tmp_path: Path):
        sections = [('## Test Section', 'Body content.')]
        md_dir = tmp_path / 'md'
        info = write_section_files(sections, md_dir)
        assert info[0]['idx'] == 1
        assert info[0]['heading'] == '## Test Section'
        assert info[0]['slug'] == 'test-section'
        assert info[0]['fname'] == '01-test-section.md'


class TestBuildIndex:
    def test_builds_index_table(self, tmp_path: Path):
        sections = [('## Intro', f'See Figure 1 for details. And Table 1 shows results. {LONG_BODY}')]
        out_dir = tmp_path
        md_dir = out_dir / 'md'
        md_dir.mkdir(parents=True)
        write_section_files(sections, md_dir)
        (out_dir / 'img' / 'sec01').mkdir(parents=True)

        n_fig, n_tbl = build_index(out_dir, md_dir, sections, {})
        assert n_fig == 1
        assert n_tbl == 1
        assert (out_dir / 'INDEX.md').exists()
        content = (out_dir / 'INDEX.md').read_text(encoding='utf-8')
        assert 'Figure 1' in content
        assert 'Table 1' in content

    def test_build_index_dedupes(self, tmp_path: Path):
        sections = [('## Intro', f'See Figure 1. And also Figure 1 again. {LONG_BODY}')]
        out_dir = tmp_path
        md_dir = out_dir / 'md'
        md_dir.mkdir(parents=True)
        write_section_files(sections, md_dir)
        (out_dir / 'img' / 'sec01').mkdir(parents=True)

        n_fig, _ = build_index(out_dir, md_dir, sections, {})
        assert n_fig == 1

    def test_build_index_with_figure_images(self, tmp_path: Path):
        sections = [('## Results', f'As shown in Figure 3. {LONG_BODY}')]
        figure_images = {'Figure 3': 'img/sec01/page-04-figure-3.png'}
        out_dir = tmp_path
        md_dir = out_dir / 'md'
        md_dir.mkdir(parents=True)
        write_section_files(sections, md_dir)

        _, _ = build_index(out_dir, md_dir, sections, figure_images)
        content = (out_dir / 'INDEX.md').read_text(encoding='utf-8')
        assert 'img/sec01/page-04-figure-3.png' in content

    def _index(self, out_dir: Path, body: str, figure_images: dict[str, str] | None = None) -> str:
        sections = [('## Shapes', body)]
        md_dir = out_dir / 'md'
        write_section_files(sections, md_dir)
        build_index(out_dir, md_dir, sections, figure_images or {})
        return (out_dir / 'INDEX.md').read_text(encoding='utf-8')

    def test_a_scanned_figure_is_the_converter_image_above_its_caption(self, tmp_path: Path):
        # A scanned page has no text layer to crop by: the image the converter placed above the caption is the figure.
        body = (
            'Figure 1 shows the core.\n\n![](../img/sec01/_page_3_Picture_4.jpeg)\n\n'
            f'Figure 1 {EN_DASH} Location of main core parts\n\nText.\n'
        )
        content = self._index(tmp_path, body)
        row = '| Figure 1 | img/sec01/_page_3_Picture_4.jpeg | md/01-shapes.md | Location of main core parts |'
        assert row in content

    def test_an_image_above_an_earlier_caption_is_not_the_next_figure(self, tmp_path: Path):
        body = (
            f'![](../img/sec01/a.jpeg)\n\nFigure 1 {EN_DASH} First\n\nFigure 2 shows nothing drawn.\n\n'
            'Figure 2 - Second, with no image of its own\n'
        )
        content = self._index(tmp_path, body)
        assert '| Figure 1 | img/sec01/a.jpeg | md/01-shapes.md | First |' in content
        assert '| Figure 2 | — | md/01-shapes.md | Second, with no image of its own |' in content

    def test_a_table_caption_is_indexed_with_its_title(self, tmp_path: Path):
        content = self._index(
            tmp_path, f'Table 3 {EN_DASH} Dimensions of RM-cores\n\n| a | b |\n|---|---|\n| 1 | 2 |\n'
        )
        assert '| Table 3 | — | md/01-shapes.md | Dimensions of RM-cores |' in content

    def test_a_cropped_render_wins_over_the_converter_image(self, tmp_path: Path):
        body = '![](../img/sec01/a.jpeg)\n\nFigure 1: First\n'
        content = self._index(tmp_path, body, {'Figure 1': 'img/sec01/page-02-figure-1.png'})
        assert '| Figure 1 | img/sec01/page-02-figure-1.png | md/01-shapes.md | First |' in content


class TestWritePaperOutput:
    def test_writes_paper_md(self, tmp_path: Path):
        sections = [
            ('', 'Some preamble text. ' * 10),
            ('## Introduction', LONG_BODY),
            ('## Conclusion', LONG_BODY),
        ]
        md_text = f'# My Paper\n\n## Introduction\n{LONG_BODY}\n\n## Conclusion\n{LONG_BODY}'
        out_dir = tmp_path
        n_sec, _n_fig, _n_tbl = write_paper_output(sections, out_dir, md_text)

        assert n_sec == 2  # Introduction + Conclusion (preamble excluded)
        assert (out_dir / 'paper.md').exists()
        content = (out_dir / 'paper.md').read_text(encoding='utf-8')
        assert 'My Paper' in content
        assert '## Abstract' in content
        assert '## Sections' in content

    def test_a_front_matter_paper_md_cannot_hold_is_a_section(self, tmp_path: Path):
        # A handbook's first pages (contents, material tables) precede its first heading: paper.md keeps only an
        # abstract-sized head of them, so the rest is a section of its own -- never dropped.
        front = 'Specific heat 750 J/(kg K). ' * 100
        sections = [('', front), ('## Introduction', LONG_BODY)]
        out_dir = tmp_path
        n_sec, _n_fig, _n_tbl = write_paper_output(sections, out_dir, f'{front}\n\n## Introduction\n{LONG_BODY}')

        assert n_sec == 2
        written = ''.join(f.read_text(encoding='utf-8') for f in sorted((out_dir / 'md').iterdir()))
        assert front.strip() in written

    def test_an_abstract_line_of_its_own_states_the_abstract(self, tmp_path: Path):
        # arXiv 2204.08094 and 2608.20674 set 'ABSTRACT' / 'Abstract' as a line over it, no heading: the abstract is
        # paper.md's, and the front matter before the first section is no section.
        abstract = 'This study analyzes the extent of the matter at hand. ' * 20
        for word in ('ABSTRACT', 'Abstract'):
            front = f'# A Paper\n\nAn Author\n\n{word}\n\n{abstract}'
            sections = [('', front), ('## 1 Introduction', LONG_BODY), ('## 2 Method', LONG_BODY)]
            out_dir = tmp_path / word
            md_text = f'{front}\n\n## 1 Introduction\n{LONG_BODY}\n\n## 2 Method\n{LONG_BODY}'
            n_sec, _n_fig, _n_tbl = write_paper_output(sections, out_dir, md_text)
            assert n_sec == 2
            assert abstract.strip() in (out_dir / 'paper.md').read_text(encoding='utf-8')

    def test_title_override(self, tmp_path: Path):
        sections = [('## Intro', LONG_BODY)]
        md_text = f'## Intro\n{LONG_BODY}'
        out_dir = tmp_path
        write_paper_output(sections, out_dir, md_text, title_override='Custom Title')
        content = (out_dir / 'paper.md').read_text(encoding='utf-8')
        assert '# Custom Title' in content

    def test_creates_img_dirs(self, tmp_path: Path):
        sections = [('## Intro', LONG_BODY)]
        md_text = f'## Intro\n{LONG_BODY}'
        out_dir = tmp_path
        write_paper_output(sections, out_dir, md_text)
        assert (out_dir / 'img' / 'flat').exists()
        assert (out_dir / 'md').exists()
