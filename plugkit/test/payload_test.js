const assert = require('assert')
const {Testing} = require('../test')

describe('Payload', () => {
  describe('#addSlice()', () => {
    it('should throw for wrong arguments', () => {
      const payload = Testing.createPayloadInstance()
      assert.throws(() => payload.addSlice(),                    TypeError)
      assert.throws(() => payload.addSlice(0),                   TypeError)
      assert.throws(() => payload.addSlice(new Uint8Array([0])), TypeError)
    })
    it('should accept an externalized Uint8Array', () => {
      const payload = Testing.createPayloadInstance()
      const array = Testing.externalize(new Uint8Array([0]))
      assert.doesNotThrow(() => payload.addSlice(array))
      assert.equal(1, payload.length)
    })
  })
  describe('#addAttr()', () => {
    it('should throw for wrong arguments', () => {
      const payload = Testing.createPayloadInstance()
      assert.throws(() => payload.addAttr(),   TypeError)
      assert.throws(() => payload.addAttr([]), TypeError)
    })
    it('should return Attr', () => {
      const payload = Testing.createPayloadInstance()
      assert.equal('Attr', payload.addAttr('.dst').constructor.name)
    })
  })
  describe('#attr()', () => {
    it('should throw for wrong arguments', () => {
      const payload = Testing.createPayloadInstance()
      assert.throws(() => payload.attr(),   TypeError)
      assert.throws(() => payload.attr([]), TypeError)
    })
    it('should return null for unknown id', () => {
      const payload = Testing.createPayloadInstance()
      assert.equal(null, payload.attr('zzz'))
      assert.equal(null, payload.attr(123))
    })
  })
  describe('#length', () => {
    it('should return payload length', () => {
      const payload = Testing.createPayloadInstance()
      const array = Testing.externalize(new Uint8Array([1, 2, 3, 4, 5]))
      assert.equal(0, payload.length)
      payload.addSlice(array)
      assert.equal(5, payload.length)
      payload.addSlice(array)
      assert.equal(10, payload.length)
    })
  })
  describe('#slices', () => {
    it('should return payload slices', () => {
      const payload = Testing.createPayloadInstance()
      const array = Testing.externalize(new Uint8Array([1, 2, 3, 4, 5]))
      assert.deepEqual([], payload.slices)
      payload.addSlice(array)
      assert.equal(1, payload.slices.length)
      payload.addSlice(array)
      assert.equal(2, payload.slices.length)
    })
  })
  describe('#attrs', () => {
    it('should return payload attrs', () => {
      const payload = Testing.createPayloadInstance()
      assert.deepEqual([], payload.attrs)
      payload.addAttr('.dst')
      assert.deepEqual(1, payload.attrs.length)
      payload.addAttr('.src')
      assert.deepEqual(2, payload.attrs.length)
    })
  })
  describe('#type', () => {
    it('should return payload type', () => {
      const payload = Testing.createPayloadInstance()
      assert.equal('', payload.type)
      payload.type = '@type'
      assert.equal('@type', payload.type)
    })
  })
  describe('#[]', () => {
    it('should return byte at given index', () => {
      const payload = Testing.createPayloadInstance()
      const array = Testing.externalize(new Uint8Array([1, 2, 3, 4, 5]))
      payload.addSlice(array)
      assert.equal(1, payload[0])
      assert.equal(2, payload[1])
      assert.equal(3, payload[2])
      assert.equal(4, payload[3])
      assert.equal(5, payload[4])
    })
    it('should return undefined when index is out of range', () => {
      const payload = Testing.createPayloadInstance()
      assert.equal(undefined, payload[0])
    })
  })
})
