"""Flask app factory.

    python -m topographer.web.app

Iteration 1's whole surface: roll a random hex, view it, reroll, step back
and forward through the rolls. No regions, no growing a map yet -- that
starts once a hex can be placed and a neighbour clicked.
"""

from __future__ import annotations

from flask import Flask

from ..pipeline.sample import default_sampler
from ..store.repo import Repo


def create_app(db_path: str = "topographer.db", cache_dir: str = ".tilecache") -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path

    # sqlite3 connections are bound to the thread that opened them, and the
    # dev server's reloader can serve a request from a different thread than
    # the one create_app() ran on -- so this is only used to apply the schema
    # once at startup. Routes open their own per-request connection via
    # flask.g (see routes.py), which is also the right pattern under a
    # production WSGI server with multiple worker threads.
    bootstrap = Repo(db_path)
    bootstrap.init_schema()
    bootstrap.close()

    app.extensions["sampler"] = default_sampler(cache_dir=cache_dir)

    from .routes import bp
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
