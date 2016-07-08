# -*- coding: utf-8 -*-
from flask import Flask, Blueprint, current_app, request, render_template, jsonify

from deepanimebot import classifiers
from deepanimebot import exceptions as exc
from deepanimebot.messages import Messages


bp = Blueprint('bp', __name__, template_folder='templates')


@bp.route('/')
def root():
    return render_template('index.html')


@bp.route('/api/v1/deepanimebot/classify_by_url')
def api_v1_classify():
    maybe_image_url = request.args.get('url')
    if maybe_image_url is None:
        return jsonify(error='provide url of image as `url` query param'), 400

    message = None
    try:
        y = current_app.extensions['classifier'].classify(url=maybe_image_url)
        return jsonify(y=y)
    except exc.TimeoutError:
        current_app.logger.debug("timed out while classifying {}".format(maybe_image_url))
        message = messages.took_too_long()
    except exc.NotImage:
        current_app.logger.debug("no image found at {}".format(maybe_image_url))
        message = messages.not_an_image()
    except Exception as e:
        current_app.logger.error("error while classifying {}: {}".format(maybe_image_url, e))
        message = messages.something_went_wrong()
    return jsonify(error=message), 500


def create_app():
    app = Flask(__name__)
    app.config.setdefault('WORKSPACE_PATH', 'default_workspace')
    app.config.setdefault('INPUT_SHAPE', 128)
    app.config.setdefault('MODEL_NAME', 'deep_anime_model')
    app.register_blueprint(bp)

    app.extensions['classifier'] = classifiers.URLClassifier(
        classifiers.ImageClassifier(
            app.config['WORKSPACE_PATH'],
            app.config['INPUT_SHAPE'],
            app.config['MODEL_NAME']))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
