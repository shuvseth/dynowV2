from PIL import Image

from dynow_vision import demo


def test_new_files_skips_seen_and_non_images(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")
    assert [p.name for p in demo.new_files(tmp_path, set())] == ["a.jpg", "b.png"]
    assert [p.name for p in demo.new_files(tmp_path, {"a.jpg"})] == ["b.png"]
    assert demo.list_images(tmp_path / "a.jpg") == [tmp_path / "a.jpg"]
