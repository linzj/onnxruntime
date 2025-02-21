// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { InferenceSession, InferenceSessionHandler, SessionHandler, Tensor } from 'onnxruntime-common';

import { Session } from './session';
import { Tensor as OnnxjsTensor } from './tensor';

export class OnnxjsSessionHandler implements InferenceSessionHandler {
  constructor(private session: Session) {
    this.inputNames = this.session.inputNames;
    this.outputNames = this.session.outputNames;
  }

  async dispose(): Promise<void> { }
  inputNames: readonly string[];
  outputNames: readonly string[];
  async run(
    feeds: SessionHandler.FeedsType,
    _fetches: SessionHandler.FetchesType,
    _options: InferenceSession.RunOptions,
  ): Promise<SessionHandler.ReturnType> {
    const inputMap = new Map<string, OnnxjsTensor>();
    for (const name in feeds) {
      if (Object.hasOwnProperty.call(feeds, name)) {
        const feed = feeds[name];
        if (feed.location === 'cpu') {
          inputMap.set(
            name,
            new OnnxjsTensor(
              feed.dims,
              feed.type as OnnxjsTensor.DataType,
              undefined,
              undefined,
              feed.data as OnnxjsTensor.NumberType,
            ),
          );
        } else if (feed.location === 'texture') {
          const onnxjsTensor = new OnnxjsTensor(
            feed.dims,
            feed.type as OnnxjsTensor.DataType,
            () => {
              throw new Error('Not implemented')
            },
            undefined,
            undefined,
          );
          onnxjsTensor.texture = feed.texture;
          onnxjsTensor.textureWidth = feed.actualTextureWidth;
          onnxjsTensor.textureHeight = feed.actualTextureHeight;
          inputMap.set(
            name,
            onnxjsTensor,
          );
        }
      }
    }
    const outputMap = await this.session.run(inputMap);
    const output: SessionHandler.ReturnType = {};
    outputMap.forEach((tensor, name) => {
      if (tensor.texture !== undefined) {
        const dims = tensor.dims;
        // Assume NCHW layout.
        output[name] = Tensor.fromTexture<"float32">(tensor.texture, { width: dims[3], height: dims[2], actualWidth: tensor.textureWidth, actualHeight: tensor.textureHeight });
      } else if (tensor.data !== undefined) {
        output[name] = new Tensor(tensor.type, tensor.data, tensor.dims);
      } else {
        throw new Error('Unexpected output type');
      }
    });
    return output;
  }
  startProfiling(): void {
    this.session.startProfiling();
  }
  endProfiling(): void {
    this.session.endProfiling();
  }
}
