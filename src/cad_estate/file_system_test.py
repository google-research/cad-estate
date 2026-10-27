# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the cached asynchronous storage client's lifetime."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from cad_estate import file_system as fs


class AsyncSessionLifetimeTest(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.sessions = []
        self.cache_patch = mock.patch.object(fs, "_gcs_async_client", None)
        self.cache_patch.start()
        self.loop_patch = mock.patch.object(fs.asyncio,
                                            "get_event_loop",
                                            return_value=self.loop)
        self.loop_patch.start()

    def tearDown(self):
        for session in self.sessions:
            if not session.closed:
                self.loop.run_until_complete(session.close())
        self.loop_patch.stop()
        self.cache_patch.stop()
        self.loop.close()

    def make_client(self):

        async def create():
            # An unused explicit token prevents credential discovery. No requests run.
            client = fs.aio_storage.Storage(token=object())
            session = client.session.session
            self.sessions.append(session)
            return client, session

        return self.loop.run_until_complete(create())

    def test_empty_context_does_not_create_a_client_or_use_the_loop(self):
        with mock.patch.object(fs.aio_storage, "Storage") as factory:
            with fs.auto_close_async_session() as value:
                self.assertIsNone(value)
            factory.assert_not_called()
        self.loop_patch.target.get_event_loop.assert_not_called()
        self.assertIsNone(fs._gcs_async_client)

    def test_existing_client_is_closed_and_removed_from_cache(self):
        client, session = self.make_client()
        fs._gcs_async_client = client
        with mock.patch.object(client, "close", wraps=client.close) as close:
            with fs.auto_close_async_session():
                self.assertIs(fs.get_gcs_async_client(), client)
                self.assertFalse(session.closed)
            close.assert_awaited_once_with()
        self.assertTrue(session.closed)
        self.assertIsNone(fs._gcs_async_client)

    def test_client_created_inside_context_is_cached_until_exit(self):
        client, session = self.make_client()
        with mock.patch.object(fs.aio_storage, "Storage",
                               return_value=client) as factory:
            with fs.auto_close_async_session():
                self.assertIs(fs.get_gcs_async_client(), client)
                self.assertIs(fs.get_gcs_async_client(), client)
                self.assertFalse(session.closed)
            factory.assert_called_once_with()
        self.assertTrue(session.closed)
        self.assertIsNone(fs._gcs_async_client)

    def test_body_exception_is_preserved_after_closing(self):
        client, session = self.make_client()
        fs._gcs_async_client = client
        failure = RuntimeError("body failed")
        with self.assertRaises(RuntimeError) as raised:
            with fs.auto_close_async_session():
                raise failure
        self.assertIs(raised.exception, failure)
        self.assertTrue(session.closed)
        self.assertIsNone(fs._gcs_async_client)

    def test_body_exception_without_a_client_is_preserved(self):
        failure = ValueError("no client was needed")
        with self.assertRaises(ValueError) as raised:
            with fs.auto_close_async_session():
                raise failure
        self.assertIs(raised.exception, failure)
        self.assertIsNone(fs._gcs_async_client)

    def test_later_context_creates_a_fresh_client(self):
        clients = [self.make_client(), self.make_client()]
        with mock.patch.object(fs.aio_storage,
                               "Storage",
                               side_effect=[pair[0] for pair in clients
                                           ]) as factory:
            for client, session in clients:
                with fs.auto_close_async_session():
                    self.assertIs(fs.get_gcs_async_client(), client)
                    self.assertFalse(session.closed)
                self.assertTrue(session.closed)
                self.assertIsNone(fs._gcs_async_client)
            self.assertEqual(factory.call_count, 2)

    def test_nested_contexts_do_not_close_twice(self):
        client, session = self.make_client()
        fs._gcs_async_client = client
        with mock.patch.object(client, "close", wraps=client.close) as close:
            with fs.auto_close_async_session():
                with fs.auto_close_async_session():
                    self.assertFalse(session.closed)
                self.assertTrue(session.closed)
                self.assertIsNone(fs._gcs_async_client)
            close.assert_awaited_once_with()

    def test_close_error_propagates_without_clearing_the_client(self):
        client, _ = self.make_client()
        fs._gcs_async_client = client
        failure = RuntimeError("close failed")
        with mock.patch.object(client, "close", side_effect=failure) as close:
            with self.assertRaises(RuntimeError) as raised:
                with fs.auto_close_async_session():
                    pass
            close.assert_awaited_once_with()
        self.assertIs(raised.exception, failure)
        self.assertIs(fs._gcs_async_client, client)

    def test_getter_reuses_existing_client_without_a_context(self):
        client, session = self.make_client()
        fs._gcs_async_client = client
        with mock.patch.object(fs.aio_storage, "Storage") as factory:
            self.assertIs(fs.get_gcs_async_client(), client)
            self.assertIs(fs.get_gcs_async_client(), client)
            factory.assert_not_called()
        self.assertFalse(session.closed)

    def test_async_local_read_write_round_trip_without_a_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "round-trip.bin")
            self.loop.run_until_complete(
                fs.write_bytes_async(path, b"round trip"))
            self.assertEqual(
                self.loop.run_until_complete(fs.read_bytes_async(path)),
                b"round trip")
            self.assertIsNone(fs._gcs_async_client)

    def test_local_async_file_read_does_not_create_a_storage_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"local contents\x00")
            with mock.patch.object(fs.aio_storage, "Storage") as factory:
                with fs.auto_close_async_session():
                    contents = self.loop.run_until_complete(
                        fs.read_bytes_async(str(path)))
                factory.assert_not_called()
            self.assertEqual(contents, b"local contents\x00")


if __name__ == "__main__":
    unittest.main()
